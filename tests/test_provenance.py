from app.provenance import load_model_manifest, validate_sources


def test_current_sources_have_explicit_review_metadata():
    result = validate_sources(production=False)
    assert result["valid"]
    assert result["sources"]
    for source in result["sources"]:
        if source["review_status"] == "approved":
            assert source["reviewer"]
            assert source["reviewed_at"]


def test_model_manifest_has_required_fingerprints():
    manifest = load_model_manifest()
    assert manifest["valid"]
    assert manifest["artifacts"]["embedding"]["artifact_sha256"]
    assert manifest["artifacts"]["reranker"]["base_model"]
