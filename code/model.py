# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import collections

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from batcher import Batcher
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS

class SummarizationModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab, batcher):
    self._hps = hps
    self._vocab = vocab
    self._batcher = batcher

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    if FLAGS.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch')
    self._target_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch')
    self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask')
    self._dec_lens = tf.placeholder(tf.int32, [hps.batch_size], name='dec_lens')

    if hps.mode=="decode" and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    if FLAGS.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed_dict[self._max_art_oovs] = batch.max_art_oovs
    if not just_enc:
      feed_dict[self._dec_batch] = batch.dec_batch
      feed_dict[self._target_batch] = batch.target_batch
      feed_dict[self._dec_padding_mask] = batch.dec_padding_mask
      feed_dict[self._dec_lens] = batch.dec_lens
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st


  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):

      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c]) # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h]) # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c) # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h) # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h) # Return new cell and state


  def _add_decoder(self, inputs, enc_padding_mask):
    """Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

    Args:
      inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

    Returns:
      outputs: List of tensors; the outputs of the decoder
      out_state: The final state of the decoder
      attn_dists: A list of tensors; the attention distributions
      p_gens: A list of scalar tensors; the generation probabilities
      coverage: A tensor, the current coverage vector
    """
    hps = self._hps
    cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

    prev_coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
    outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(
      inputs, self._dec_in_state, self._enc_states, enc_padding_mask, cell, initial_state_attention=(hps.mode=="decode"),
      pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=prev_coverage,
      feedback=None, max_dec_steps=hps.max_dec_steps)

    return outputs, out_state, attn_dists, p_gens, coverage
  
  def _calc_final_dist(self, vocab_dists, attn_dists, p_gens):
    return self._calc_final_dist_extend_vocab(vocab_dists, attn_dists, p_gens, self._enc_batch_extend_vocab)  

  def _calc_final_dist_extend_vocab(self, vocab_dists, attn_dists, p_gens, enc_batch_extend_vocab):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    batch_size = int(vocab_dists[0].shape[0])
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words (i-th OOV word of article has id of (vocab size + i))
      extended_vsize = self._vocab.size() + Batcher.MAX_OOVS # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((batch_size, Batcher.MAX_OOVS))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists

  def _calc_attended_emb(self, emb_enc_inputs, attn_dists):
    attn_dists_e = tf.expand_dims(attn_dists, axis=3)  # (batch_size, <dec timesteps>, <enc length>, 1)
    emb_enc_inputs_e = tf.expand_dims(emb_enc_inputs, axis=1)  # (batch_size, 1, <enc length>, emb_dim)
    # Return reduced across timesteps, shape (batch_size, <dec timesteps>, emb_dim)
    return tf.reduce_sum(attn_dists_e * emb_enc_inputs_e, axis=2)

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  Beam = collections.namedtuple("Beam", ("logprobs", "lens", "is_not_stopped", "output_vocabs", "outputs", "attn_dists", "p_gens", "states", "coverage"))
  def expand_and_reduce_beam(self, beam, k, output_beam_width,
      outputs_step, out_state, attn_dists, p_gens, coverage, output_dists_extend_vocab):
    """
    Given beam in tile_batch order (batch_size * output_beam_width, ...),
    extends outputs using the top K next words for each beam,
    then reduces back down to top beam_width extended candidates for each batch member.
    Returns an updated beam in tile_batch order.
    When decoding output_beam_width should be beam_width except in the last step when we pick one last candidate.
    """
    hps = self._hps
    beam_width = outputs_step[0].shape[0] // hps.batch_size    
    assert len(outputs_step) == 1
    assert len(attn_dists) == 1
    assert len(p_gens) == 1
    next_logprobs = []
    next_states_c = []
    next_states_h = []
    next_beam_output_vocabs = []
    next_beam_outputs = []
    next_beam_attn_dists = []
    next_beam_p_gens = []
    next_coverage = []
    next_is_not_stopped = []
    next_lens = []
    stop_decode = self._batcher._stop_decoding_id
    # Get top beam_width candidates (among beam_width * K) for each element in batch.

    for batch_i in range(hps.batch_size):
      # Indexes for this batch member
      (member_l, member_r) = (batch_i * beam_width, (batch_i + 1) * beam_width)

      # Extend each beam with top K next words.
      topk_probs, topk_vocab = tf.nn.top_k(output_dists_extend_vocab[member_l:member_r], k)  # topk_probs/vocab has shape (beam_width, k)

      # Compute averages. Stopped sentences do not get incremental logprobs/lens, so their avg logprob remains the same always.
      cumulative_probs = tf.expand_dims(beam.logprobs[member_l:member_r], 1) + tf.log(topk_probs) * tf.expand_dims(tf.cast(beam.is_not_stopped[member_l:member_r], dtype=tf.float32), 1)
      sent_lens = beam.lens[member_l:member_r] + 1 * beam.is_not_stopped[member_l:member_r]
      avg_logprobs = cumulative_probs / tf.cast(tf.expand_dims(sent_lens, 1), tf.float32)

      # Pick output_beam_width best among the (beam_width * K) candidates.
      _probs, topk_candidate_indices = tf.nn.top_k(tf.reshape(avg_logprobs, [beam_width * k]), output_beam_width)  # (output_beam_width) ids/probs
      chosen_vocabs = tf.gather(tf.reshape(topk_vocab, [beam_width * k]), topk_candidate_indices)  # gathered shape (beam_width)

      # Indices back into the source rays whose next words won.
      source_beam_indices = topk_candidate_indices // k

      # Update lens and probs. Note that we ranked by avg logprobs but we save total probs and len separately.
      next_lens.append(tf.gather(sent_lens, source_beam_indices))
      
      winner_probs = tf.gather(tf.reshape(topk_probs, [beam_width * k]), topk_candidate_indices)
      next_logprobs.append(
        tf.gather(beam.logprobs[member_l:member_r], source_beam_indices) +
        tf.log(winner_probs) * tf.gather(tf.cast(beam.is_not_stopped[member_l:member_r], dtype=tf.float32), source_beam_indices))

      # Change output_vocabs to stop_decode after any stop_decode. This is for debuggining only. We zero out outputs for recode later.
      source_is_not_stopped = tf.gather(beam.is_not_stopped[member_l:member_r], source_beam_indices)
      zerod_chosen_vocabs = [] 
      for i in range(chosen_vocabs.shape[0]):
        zerod_chosen_vocabs.append(tf.cond(tf.equal(source_is_not_stopped[i], 0), lambda: stop_decode, lambda: chosen_vocabs[i]))
      
      # Attach the next words to the corresponding source beams.
      next_beam_output_vocabs.append(tf.concat([
        tf.gather(beam.output_vocabs[member_l:member_r], source_beam_indices),
        tf.expand_dims(zerod_chosen_vocabs, axis=1)], axis=1))

      # Carry over info corresponding to the chosen source beams.
      next_states_c.append(tf.gather(out_state.c[member_l:member_r], source_beam_indices))
      next_states_h.append(tf.gather(out_state.h[member_l:member_r], source_beam_indices))
      if hps.coverage:
        next_coverage.append(tf.gather(coverage[member_l:member_r], source_beam_indices))
      next_beam_outputs.append(tf.concat([
        tf.gather(beam.outputs[member_l:member_r], source_beam_indices),
        tf.gather(tf.transpose(outputs_step[-1:], [1, 0, 2])[member_l:member_r], source_beam_indices)], axis=1))
      if beam.attn_dists is None:
        next_beam_attn_dists.append(tf.gather(tf.transpose(attn_dists[-1:], [1, 0, 2])[member_l:member_r], source_beam_indices))
      else:
        next_beam_attn_dists.append(tf.concat([
          tf.gather(beam.attn_dists[member_l:member_r], source_beam_indices),
          tf.gather(tf.transpose(attn_dists[-1:], [1, 0, 2])[member_l:member_r], source_beam_indices)], axis=1))
      next_beam_p_gens.append(tf.concat([
        tf.gather(beam.p_gens[member_l:member_r], source_beam_indices),
        tf.gather(tf.transpose(p_gens[-1:], [1, 0, 2])[member_l:member_r], source_beam_indices)], axis=1))            

      # Update is_not_stopped by checking if any stop_decoding = 3 symbols
      for i in range(source_is_not_stopped.shape[0]):
        is_still_not_stopped = tf.cond(tf.equal(chosen_vocabs[i], stop_decode), lambda: 0, lambda: 1)
        next_is_not_stopped.append(source_is_not_stopped[i] * is_still_not_stopped)
    next_is_not_stopped = tf.stack(next_is_not_stopped, axis=0)

    # Now concat them. They are listed clustered in batch order so the result is in tile_batch order.
    new_beam = self.Beam(
      states = tf.contrib.rnn.LSTMStateTuple(c=tf.concat(next_states_c, axis=0), h=tf.concat(next_states_h, axis=0)),
      logprobs = tf.concat(next_logprobs, axis=0),
      lens = tf.concat(next_lens, axis=0),
      is_not_stopped = next_is_not_stopped,
      output_vocabs = tf.concat(next_beam_output_vocabs, axis=0),
      outputs = tf.concat(next_beam_outputs, axis=0),
      attn_dists = tf.concat(next_beam_attn_dists, axis=0),
      coverage = tf.concat(next_coverage, axis=0) if hps.coverage else None,
      p_gens = tf.concat(next_beam_p_gens, axis=0))
    assert new_beam.logprobs.shape[0] == hps.batch_size * output_beam_width
    assert new_beam.states.c.shape[0] == hps.batch_size * output_beam_width
    assert new_beam.lens.shape[0] == hps.batch_size * output_beam_width
    assert new_beam.is_not_stopped.shape[0] == hps.batch_size * output_beam_width
    assert new_beam.output_vocabs.shape[0] == hps.batch_size * output_beam_width
    assert new_beam.attn_dists.shape[0] == hps.batch_size * output_beam_width
    if hps.coverage:
      assert new_beam.coverage.shape[0] == hps.batch_size * output_beam_width
    assert new_beam.p_gens.shape[0] == hps.batch_size * output_beam_width
    return new_beam


  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init, trainable=hps.trainable_embedding)        
        if hps.mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)

      # Add the encoder.
      enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)

      self._enc_states = enc_outputs

      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
      self._dec_in_state = self._reduce_states(fw_st, bw_st)
      
      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [hps.hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
        v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)              

      # Add the decoder.
      with tf.variable_scope('decoder') as dec_scope:
        cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)
        
        USE_BEAM = FLAGS.use_beam and (hps.mode != 'decode' or FLAGS.beam_decode)
        if USE_BEAM:
          BEAM_WIDTH = 4
          K = BEAM_WIDTH
          # Everything below is in tile_batch order.
          # These are accumulated across timesteps.
          beam = self.Beam(
            output_vocabs = tf.zeros([hps.batch_size, 0], dtype=tf.int32), # (batch_size * BEAM_WIDTH, timesteps) giving the output ids
            outputs = tf.zeros([hps.batch_size, 0, hps.hidden_dim]),  # the outputs
            attn_dists = None,  # This should have shape (batch_size * BEAM_WIDTH, 0, <enc input length>) but we don't know length of input.
            coverage = None,
            p_gens = tf.zeros([hps.batch_size, 0, 1]),  # the outputs
            # These are updated in place each timestep.
            logprobs = tf.zeros([hps.batch_size]),
            lens = tf.zeros([hps.batch_size], dtype=tf.int32),
            is_not_stopped = tf.ones([hps.batch_size], dtype=tf.int32),
            states = self._dec_in_state)  # each c/h has shape (batch_size * BEAM_WIDTH, hidden_dim)

          last_outputs = emb_dec_inputs[0]
          enc_states = self._enc_states
          enc_padding_mask = self._enc_padding_mask
          enc_batch_extend_vocab = self._enc_batch_extend_vocab
    
          BEAM_DECODE_STEPS = hps.beam_max_dec_steps if hps.mode == 'decode' else len(emb_dec_inputs)
          step = 0
          while step < BEAM_DECODE_STEPS:
            beam_width = beam.lens.shape[0] // self._hps.batch_size
            if step == 1:
              # Widen once after first step.
              enc_states = tf.contrib.seq2seq.tile_batch(self._enc_states, beam_width)
              enc_padding_mask = tf.contrib.seq2seq.tile_batch(self._enc_padding_mask, beam_width)
              enc_batch_extend_vocab = tf.contrib.seq2seq.tile_batch(self._enc_batch_extend_vocab, beam_width)
  
            with tf.variable_scope(dec_scope, reuse=tf.AUTO_REUSE):
              # decode one step
              outputs_step, out_state, _states, attn_dists, p_gens, coverage_step = attention_decoder(
                [last_outputs], beam.states, enc_states, enc_padding_mask, cell, initial_state_attention=True,
                pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=beam.coverage,
                steps=1)

            output_vocab_softmax = [ tf.nn.softmax(tf.nn.xw_plus_b(outputs_step[-1], w, v)) ]  # (batch_size * BEAM_WIDTH, vsize)
            output_dists_extend_vocab = self._calc_final_dist_extend_vocab(output_vocab_softmax, attn_dists, p_gens, enc_batch_extend_vocab)[0]

            output_beam_width = BEAM_WIDTH if step + 1 < BEAM_DECODE_STEPS else 1
            beam = self.expand_and_reduce_beam(
              beam, K, output_beam_width,
              outputs_step, out_state, attn_dists, p_gens, coverage_step, output_dists_extend_vocab)
            oov_to_unk = tf.minimum(beam.output_vocabs[:,-1], vsize)
            last_outputs = tf.nn.embedding_lookup(tf.concat([embedding, embedding[0:1]], axis=0), oov_to_unk)

            step += 1
        
        # Now the regular teacher forced decode.
        # Note that in abisee's original code there was a mismatch between training and decode where initial state attention is used in decode but
        # not in training so the first context vector in training is just zeros, unlike in decode. We avoid a mismatch here by always using initial_state_attention.
        coverage = self.prev_coverage if hps.mode=="decode" and hps.coverage else None # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time        
        with tf.variable_scope(dec_scope, reuse=tf.AUTO_REUSE):
          decoder_outputs, self._dec_out_state, _states, self.attn_dists, self.p_gens, self.coverage= attention_decoder(
            emb_dec_inputs, self._dec_in_state, self._enc_states, self._enc_padding_mask, cell, initial_state_attention=True,
            pointer_gen=hps.pointer_gen, use_coverage=hps.coverage, prev_coverage=coverage,
            steps=len(emb_dec_inputs))

      self._dec_outputs = decoder_outputs
      
      # Sanity checks
      assert len(decoder_outputs) == len(emb_dec_inputs)

      # Add the output projection to obtain the vocabulary distribution. This is V', b' in the paper.
      vocab_scores = [] # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
      for _i,output in enumerate(decoder_outputs):
        vocab_scores.append(tf.nn.xw_plus_b(output, w, v)) # apply the linear layer

      vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
      if FLAGS.pointer_gen:
        final_dists = self._calc_final_dist(vocab_dists, self.attn_dists, self.p_gens)
        tf.summary.scalar('p_gens', tf.reduce_mean(tf.reduce_sum(tf.squeeze(tf.stack(self.p_gens, axis=1), axis=2) * self._dec_padding_mask, axis=1) / tf.cast(self._dec_lens, tf.float32)))        
      else: # final distribution is just vocabulary distribution
        final_dists = vocab_dists

      def recode(beam_outputs, beam_lens, beam_output_vocabs, beam_mask):
        with tf.variable_scope("subn"):
          recode_dim = 128
          recode_inputs = tf.map_fn(lambda output: tf.matmul(tf.nn.softmax(tf.nn.xw_plus_b(output, w, v), axis=1), embedding), beam_outputs)
          cell_fw = tf.contrib.rnn.GRUCell(recode_dim, kernel_initializer=self.rand_unif_init, bias_initializer=self.rand_unif_init)
          cell_bw = tf.contrib.rnn.GRUCell(recode_dim, kernel_initializer=self.rand_unif_init, bias_initializer=self.rand_unif_init)
          (re_encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, recode_inputs, sequence_length=beam_lens, dtype=tf.float32, swap_memory=True)
          re_encoder_outputs = tf.concat(axis=2, values=re_encoder_outputs) # concatenate the forwards and backwards states, (batch_size, 2 * hidden_dim)
          re_enc_final = tf.concat([fw_st, bw_st], axis=-1)
          re_in_state = tf.contrib.rnn.LSTMStateTuple(c=tf.layers.dense(re_enc_final, recode_dim, name="c"), h=tf.layers.dense(re_enc_final, recode_dim, name="h"))
          re_cell = tf.contrib.rnn.LSTMCell(recode_dim, state_is_tuple=True, initializer=self.rand_unif_init)
          re_decoder_outputs, _dec_out_state, _states, re_attn_dists, re_p_gens, _re_coverage = attention_decoder(
            emb_dec_inputs, re_in_state, re_encoder_outputs, beam_mask, re_cell, initial_state_attention=False,
            pointer_gen=True, use_coverage=True,
            steps=len(emb_dec_inputs))
          recode_output = tf.layers.dense(tf.stack(re_decoder_outputs), hps.hidden_dim, name="recode_outputs")
          re_vocab_scores = [ tf.nn.softmax(tf.nn.xw_plus_b(output, w, v)) for output in tf.unstack(recode_output) ]
          tf.summary.scalar('re_p_gens', tf.reduce_mean(tf.reduce_sum(tf.squeeze(tf.stack(re_p_gens, axis=1), axis=2) * self._dec_padding_mask, axis=1) / tf.cast(self._dec_lens, tf.float32)))
          return (self._calc_final_dist_extend_vocab(re_vocab_scores, re_attn_dists, re_p_gens, beam_output_vocabs), re_attn_dists)

      # Re-encode/decode first_output
      if USE_BEAM and hps.mode != 'decode':
        USE_RECODE = FLAGS.use_recode

        def len_to_mask(length, dim):
          """
          Return a tensor of (dim) with len 1's followed by 0's
          """
          bits = []
          for i in range(dim):
            bits.append(tf.cond(tf.less(i, length), lambda: 1.0, lambda: 0.0))
          return tf.stack(bits)

        beam_mask = tf.map_fn(lambda row: len_to_mask(row, beam.outputs.shape[1]), beam.lens, dtype=tf.float32)  # (batch_size, max_dec_len)
        tf.summary.scalar('beam_p_gens', tf.reduce_mean(
          tf.reduce_sum(tf.squeeze(beam.p_gens, axis=2) * beam_mask, axis=1) / tf.cast(beam.lens, dtype=tf.float32)))
        if USE_RECODE:
          (re_final_dists, re_attn_dists) = recode(beam.outputs, beam.lens, beam.output_vocabs, beam_mask)

        # There's a loss for every word after len.
        word_loss_mask = tf.expand_dims(tf.ones((vsize,)) - tf.one_hot(3, vsize), axis=0)  # (1, vsize) where only stop_decode is 0 rather than 1
        beam_losses = []
        dec_lens = tf.reduce_sum(self._dec_padding_mask, axis=1)  # (batch_size)
        for (i, output) in enumerate(tf.unstack(beam.outputs, axis=1)):
          overage = tf.maximum(tf.cast(i, dtype=tf.float32) - 0.8 * dec_lens, 0.0)
          penalty = tf.pow(1.04, overage) - 1.0
          word_output = tf.nn.softmax(tf.nn.xw_plus_b(output, w, v), axis=1)  # (batch_size, vsize)
          word_loss = penalty * tf.reduce_sum(word_output * word_loss_mask, axis=-1)  # (batch_size)
          beam_losses.append(word_loss)
        beam_losses_stacked = tf.stack(beam_losses, axis=1)  # (batch_size, dec len)
        beam_losses_masked = beam_losses_stacked * beam_mask
        per_word_losses = tf.reduce_sum(beam_losses_masked, axis=1) / int(beam.outputs.shape[1])  # shape (batch_size)
        len_loss = tf.reduce_mean(per_word_losses)
        if hps.len_loss_wt != 0:
          tf.summary.scalar('len_loss', len_loss)

      if hps.mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          if FLAGS.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
            loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
            
            batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
            for dec_step, dist in enumerate(final_dists):
              targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
              indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)

              gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
              losses = -tf.log(gold_probs)

              # Try this line below if you get inf/nan
              GOT_INFS = True
              if GOT_INFS:
                losses = -tf.log(tf.maximum(gold_probs, tf.ones(gold_probs.get_shape()) * 1e-10))
                              
              loss_per_step.append(losses)

            loss_output = _mask_and_avg(loss_per_step, self._dec_padding_mask)
            
            if USE_BEAM and USE_RECODE:
              re_loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)              
              for dec_step, dist in enumerate(re_final_dists):
                targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
                indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
                re_gold_probs = tf.gather_nd(re_final_dists[dec_step], indices)
                re_losses = -tf.log(re_gold_probs)
                # Try this line below if you get inf/nan
                GOT_INFS = True
                if GOT_INFS:
                  re_losses = -tf.log(tf.maximum(re_gold_probs, tf.ones(re_gold_probs.get_shape()) * 1e-10))
                re_loss_per_step.append(re_losses)
              loss_recode = _mask_and_avg(re_loss_per_step, self._dec_padding_mask)
          else: # baseline model
            loss_output = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, self._dec_padding_mask) # this applies softmax internally

          if USE_BEAM:
            tf.summary.scalar('len', tf.reduce_mean(beam.lens))
            if USE_RECODE:
              tf.summary.scalar('loss_recode', loss_recode)
              self._loss = loss_output + loss_recode
            else:
              self._loss = loss_output
            self._loss = self._loss +  + hps.len_loss_wt * len_loss
          else:
            self._loss = loss_output
          tf.summary.scalar('loss', self._loss)
          tf.summary.scalar('loss_output', loss_output)

          # Calculate coverage loss from the attention distributions
          if hps.coverage:
            self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
            tf.summary.scalar('coverage_loss', self._coverage_loss)                            
            with tf.variable_scope('coverage_loss'):
              if USE_BEAM:
                self._beam_coverage_loss = _coverage_loss(tf.unstack(beam.attn_dists, axis=1), beam_mask)
                tf.summary.scalar('beam_coverage_loss', self._beam_coverage_loss)
                if USE_RECODE:
                  self._recode_coverage_loss = _coverage_loss(re_attn_dists, self._dec_padding_mask)
                  tf.summary.scalar('recode_coverage_loss', self._recode_coverage_loss)
                self._total_loss = self._loss + hps.cov_loss_wt * (self._coverage_loss + self._beam_coverage_loss)
              else:
                self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
            tf.summary.scalar('total_loss', self._total_loss)


    if hps.mode == "decode":
      # We run decode beam search mode one decoder step at a time
      assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
      final_dists = final_dists[0]
      topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
      self._topk_log_probs = tf.log(topk_probs)

      if USE_BEAM:
        self._first_ids = beam.output_vocabs

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)

    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, batch):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(batch, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder

    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])

    return enc_states, dec_in_state


  def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage, return_first=False):
    """For beam search decoding. Run the decoder for one step.

    Args:
      sess: Tensorflow session.
      batch: Batch object containing single example repeated across the batch
      latest_tokens: Tokens to be fed as input into the decoder for this timestep
      enc_states: The encoder states.
      dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
      prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.

    Returns:
      ids: top 2k ids. shape [beam_size, 2*beam_size]
      probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
      new_states: new states of the decoder. a list length beam_size containing
        LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
      attn_dists: List length beam_size containing lists length attn_length.
      p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
      new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
    """

    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_in_state: new_dec_in_state,
        self._dec_batch: np.transpose(np.array([latest_tokens])),
        self._enc_batch: batch.enc_batch,
        self._enc_lens: batch.enc_lens,
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "dec_outputs": self._dec_outputs,
      "attn_dists": self.attn_dists
    }
    if return_first:
      to_return["first_ids"] = self._first_ids

    if FLAGS.pointer_gen:
      feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
      feed[self._max_art_oovs] = batch.max_art_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in range(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()

    if FLAGS.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in range(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if FLAGS.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in range(beam_size)]

    first_ids = results['first_ids'] if return_first else None 
    return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage, first_ids


def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average


def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  i = 0
  for a in attn_dists:
    i += 1
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss
