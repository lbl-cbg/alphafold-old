import haiku as hk
import jax.numpy as jnp

class ResidualAdjustment(hk.Module):

  def __init__(self, config, global_config, name='residual_adjust'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config


  def __call__(self, representations, prob_r, is_training):
    """Builds ResiduaAdjustment module.

    Arguments:
      representations: Dictionary of representations, must contain
        * 'pair': pair representation, shape [N_res, N_res, c_z]
      prob_r: P(r) curves, shape [N_bins]
      is_training: Whether the module is in training mode.

    Returns:

    """
    distogram = DistogramHead(
        self.config,
        self.global_config)(
            representations, batch, is_training)

    collapsed_distogram = jnp.einsum('ijkl->il', batch)

    collapsed_distogram = common_modules.Linear(
        self.config.num_hidden_residuals,
        initializer=utils.final_init(self.global_config), # TODO: figure out an appropriate initializer
        name='coll_dist_linear')(
            collapsed_distogram)

    prob_r = common_modules.Linear(
        self.config.num_hidden_residuals,
        initializer=utils.final_init(self.global_config), # TODO: figure out an appropriate initializer
        name='prob_r_linear')(
            prob_r)

    residuals = prob_r - collapsed_distogram

   representations['msa'], residuals




def softmax_cross_entropy(logits, labels):
  """Computes softmax cross entropy given logits and one-hot class labels."""
  loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
  return jnp.asarray(loss)


class DistogramHead(hk.Module):
  """Head to predict a distogram.

  Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
  """

  def __init__(self, config, global_config, name='distogram_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self, representations, batch, is_training):
    """Builds DistogramHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * logits: logits for distogram, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
    """
    half_logits = common_modules.Linear(
        self.config.num_bins,
        initializer=utils.final_init(self.global_config),
        name='half_logits')(
            representations['pair'])

    logits = half_logits + jnp.swapaxes(half_logits, -2, -3)
    breaks = jnp.linspace(self.config.first_break, self.config.last_break,
                          self.config.num_bins - 1)

    return dict(logits=logits, bin_edges=breaks)

  def loss(self, value, batch):
    return _distogram_log_loss(value['logits'], value['bin_edges'],
                               batch, self.config.num_bins)


def _distogram_log_loss(logits, bin_edges, batch, num_bins):
  """Log loss of a distogram."""

  assert len(logits.shape) == 3
  positions = batch['pseudo_beta']
  mask = batch['pseudo_beta_mask']

  assert positions.shape[-1] == 3

  sq_breaks = jnp.square(bin_edges)

  dist2 = jnp.sum(
      jnp.square(
          jnp.expand_dims(positions, axis=-2) -
          jnp.expand_dims(positions, axis=-3)),
      axis=-1,
      keepdims=True)

  true_bins = jnp.sum(dist2 > sq_breaks, axis=-1)

  errors = softmax_cross_entropy(
      labels=jax.nn.one_hot(true_bins, num_bins), logits=logits)

  square_mask = jnp.expand_dims(mask, axis=-2) * jnp.expand_dims(mask, axis=-1)

  avg_error = (
      jnp.sum(errors * square_mask, axis=(-2, -1)) /
      (1e-6 + jnp.sum(square_mask, axis=(-2, -1))))
  dist2 = dist2[..., 0]
  return dict(loss=avg_error, true_dist=jnp.sqrt(1e-6 + dist2))


class Attention(hk.Module):
  """Multihead attention."""

  def __init__(self, config, global_config, output_dim, name='attention'):
    super().__init__(name=name)

    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim

  def __call__(self, q_data, m_data, mask, nonbatched_bias=None):
    """Builds Attention module.

    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      mask: A mask for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].

    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    # Sensible default for when the config keys are missing
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    assert key_dim % num_head == 0
    assert value_dim % num_head == 0
    key_dim = key_dim // num_head
    value_dim = value_dim // num_head

    q_weights = hk.get_parameter(
        'query_w', shape=(q_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    k_weights = hk.get_parameter(
        'key_w', shape=(m_data.shape[-1], num_head, key_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())
    v_weights = hk.get_parameter(
        'value_w', shape=(m_data.shape[-1], num_head, value_dim),
        dtype=q_data.dtype,
        init=glorot_uniform())

    q = jnp.einsum('bqa,ahc->bqhc', q_data, q_weights) * key_dim**(-0.5)
    k = jnp.einsum('bka,ahc->bkhc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bkhc', m_data, v_weights)
    logits = jnp.einsum('bqhc,bkhc->bhqk', q, k)
    if nonbatched_bias is not None:
      logits += jnp.expand_dims(nonbatched_bias, axis=0)
    logits = jnp.where(mask, logits, _SOFTMAX_MASK)
    weights = utils.stable_softmax(logits)
    weighted_avg = jnp.einsum('bhqk,bkhc->bqhc', weights, v)

    if self.global_config.zero_init:
      init = hk.initializers.Constant(0.0)
    else:
      init = glorot_uniform()

    if self.config.gating:
      gating_weights = hk.get_parameter(
          'gating_w',
          shape=(q_data.shape[-1], num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(0.0))
      gating_bias = hk.get_parameter(
          'gating_b',
          shape=(num_head, value_dim),
          dtype=q_data.dtype,
          init=hk.initializers.Constant(1.0))

      gate_values = jnp.einsum('bqc, chv->bqhv', q_data,
                               gating_weights) + gating_bias

      gate_values = jax.nn.sigmoid(gate_values)

      weighted_avg *= gate_values

    o_weights = hk.get_parameter(
        'output_w', shape=(num_head, value_dim, self.output_dim),
        dtype=q_data.dtype,
        init=init)
    o_bias = hk.get_parameter(
        'output_b', shape=(self.output_dim,),
        dtype=q_data.dtype,
        init=hk.initializers.Constant(0.0))

    output = jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias

    return output

