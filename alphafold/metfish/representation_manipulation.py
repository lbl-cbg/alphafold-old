import numpy as np


def reinitialize_representations(prev, aatype, multimer_mode):
    """Resets the recycling inputs

    Reinitializes the pair, position (structural), and MSA (first row only) representations
    that are used as inputs for the next recycling iteration of AlphaFold

    Args:
      prev: Dict of pair, position, and msa representations (currently not used)
      aatype: Amino acid type, given as array with integers.
      multimer_mode: Bool indicating whether currently using multimer mode
    Returns:
      dict of pair, position, and msa representations all set to zero
    """

    if multimer_mode:
        L = aatype.shape[0]
    else:
        L = aatype.shape[1]

    # reinitialize
    zeros = lambda shape: np.zeros(shape, dtype=np.float16)
    prev = {'prev_msa_first_row': zeros([L, 256]),
            'prev_pair': zeros([L, L, 128]),
            'prev_pos': zeros([L, 37, 3])}

    return prev