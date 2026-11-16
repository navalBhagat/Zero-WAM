import json
from pathlib import Path

import pytest
from evaluation.robotwin.demonstration_retrieval import DemonstrationRetriever


def test_retrieval_uses_scene_count_caption_metadata_and_excludes_other_tasks(tmp_path):
    def add(name, caption=None):
        directory=tmp_path/name;directory.mkdir()
        path=directory/'generated_video_kling-v3.pth';path.touch()  # Must not deserialize latent files.
        if caption:
            (directory/'metadata.json').write_text(json.dumps({'human_caption': caption}))
        return str(path.resolve())
    single=add('001_robotwin_place_one_bread_in_basket')
    double=add('002_robotwin_place_two_breads_in_basket')
    scale=add('003_robotwin_generic', 'Place an object onto the electronic scale')
    retriever=DemonstrationRetriever(tmp_path)
    assert retriever.select('place_bread_basket','Place bread in basket',1)[1]==single
    assert retriever.select('place_bread_basket','Place bread in basket',2)[1]==double
    _, selected, audit=retriever.select('place_object_scale','Put the bell on the scale')
    assert selected==scale and audit['candidate_count']==1
    assert audit['selected']['caption_source'].endswith('metadata.json:human_caption')
    assert retriever.select('place_object_scale','Put the bell on the scale')[2]==audit
    with pytest.raises(ValueError,match='No generated_video'):
        DemonstrationRetriever(tmp_path/'missing')
