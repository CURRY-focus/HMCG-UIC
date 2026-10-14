"""Public model entry point for the complete HMCG-UIC method."""

from stablecodec_backbone import StableCodecBackbone


class HMCGUIC(StableCodecBackbone):
    """Human-machine collaborative codec described in the manuscript.

    The StableCodec-derived reconstruction backbone is deliberately retained
    in :mod:`stablecodec_backbone`.  ``StableCodecBackbone`` instantiates the
    HMCG-UIC latent codec, so this paper-facing class has exactly the same
    training, compression, and decompression behavior while making the
    upstream inheritance explicit.
    """

    pass
