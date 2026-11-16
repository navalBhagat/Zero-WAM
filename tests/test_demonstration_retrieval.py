import json
from pathlib import Path

import pytest
import torch
from evaluation.robotwin.demonstration_retrieval import DemonstrationRetriever


def test_retrieval_uses_scene_count_caption_metadata_and_excludes_other_tasks(tmp_path):
    def add(name, caption=None):
        directory=tmp_path/name;directory.mkdir()
        path=directory/'generated_video_kling-v3.pth'
        torch.save({'latent': torch.zeros(1)}, path)
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


def test_cached_caption_overrides_misleading_folder_and_ignores_hand_count(tmp_path):
    from evaluation.robotwin.demonstration_retrieval import bread_count
    names=['007_robotwin_If_there_is_one_bread_on_the_table',
           '008_robotwin_If_there_is_one_bread_on_the_table']
    captions=['Simultaneously use both hands to pick up an orange bread from the left and a yellow bread from the right, and place them into a bowl.',
              'Use both hands to place a yellow bread into the basket.']
    for name,caption in zip(names,captions):
        path=tmp_path/name;path.mkdir()
        torch.save({'text':caption,'task':'Only two human arms are permitted.',
                    'latent':torch.zeros(2,48),'text_emb':torch.zeros(3,4)},path/'generated_video.pth')
    r=DemonstrationRetriever(tmp_path)
    _,latent,audit=r.select('place_bread_basket','Place the yellow bread in the basket',1)
    assert names[1] in latent
    assert audit['selected']['caption_source'].endswith('.pth:text')
    assert bread_count(captions[0])==2 and bread_count(captions[1])==1
    assert r.select('place_bread_basket','Place both breads in the basket',2)[1].endswith(names[0]+'/generated_video.pth')
