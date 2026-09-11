import pytest
import torch

from miracle_ad.models import (
    available_backbones,
    available_networks,
    create_backbone,
    get_model,
)


EXPECTED_BACKBONES = {
    "kurdish_cnn",
    "gamma_erb_cnn",
    "gamma_gm_cnn",
    "soundnet8",
    "soundnet5",
    "m3",
    "m5",
    "m11",
    "m18",
    "raw_audio_cnn",
    "wavenet",
    "raw_audio_lstm",
    "wav2vec2",
}


def test_all_research_backbone_options_are_retained():
    assert set(available_backbones()) == EXPECTED_BACKBONES


@pytest.mark.parametrize("backbone", sorted(EXPECTED_BACKBONES - {"wav2vec2"}))
def test_local_backbone_options_forward(backbone):
    model = create_backbone(backbone)
    output = model(torch.randn(1, 1, 24000))
    assert output.shape == (1, model.feature_dim)


@pytest.mark.parametrize("backbone", ["gamma_gm_cnn", "gamma_erb_cnn"])
def test_primary_gammatone_backbones_forward(backbone):
    model = get_model(
        backbone_type=backbone,
        network_type="abmil",
        num_disease_classes=3,
        lang_aware=False,
    )
    output = model(torch.randn(1, 2, 24000))
    assert output.shape == (1, 3)


@pytest.mark.parametrize("network", available_networks())
def test_pooling_options_support_four_language_head(network):
    model = get_model(
        backbone_type="m3",
        network_type=network,
        num_disease_classes=3,
        num_language_classes=4,
        lang_aware=True,
    )
    disease, language = model(torch.randn(1, 2, 24000))
    assert disease.shape == (1, 3)
    assert language.shape == (1, 4)


@pytest.mark.parametrize("network", ["abmil", "gated_abmil", "transformer_abmil"])
def test_attention_pooling_returns_normalized_chunk_weights(network):
    model = get_model(
        backbone_type="m3",
        network_type=network,
        num_disease_classes=3,
        num_language_classes=4,
        lang_aware=True,
    )
    (disease, language), attention = model(
        torch.randn(1, 3, 24000),
        return_attention=True,
    )
    assert disease.shape == (1, 3)
    assert language.shape == (1, 4)
    assert attention.shape == (1, 3)
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(1))
