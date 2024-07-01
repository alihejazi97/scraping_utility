from general_utility import extract_segment_from_array
import pysubs2
import  torch
from torch.nn import Module
import numpy as np
from scraping_utility.settings import Settings, default_setting
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from typing import  Iterable


def compute_similarity(audio_array: np.array, sample_rate: int, subtitle: pysubs2.SSAEvent,
                       speech_embedding_model: SpeechToEmbeddingModelPipeline, text_embedding_model: TextToEmbeddingModelPipeline):
    _audio_array = extract_segment_from_array(audio_array, sample_rate, subtitle.text)
    _speech_embedding = speech_embedding_model.predict([torch.Tensor(_audio_array).unsqueeze(dim=0)])
    _text_embedding = text_embedding_model.predict([subtitle.content], source_lang='pes_Arab')
    return Settings.SIMILARITY_MEASURE(_speech_embedding, _text_embedding)

@default_setting([('speech_embedding_batch_size' , 5, 'SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE',),])
def compute_batch_speech_embedding(audio_array: np.array, sample_rate: float, subtitle_list: Iterable[pysubs2.SSAEvent],
                                   speech2embedding_model:SpeechToEmbeddingModelPipeline, speech_embedding_batch_size: int=None):
    _audio_array_list = []
    for _subtitle in subtitle_list:
        _audio_array = extract_segment_from_array(audio_array, sample_rate, _subtitle)
        _audio_array_list.append(torch.Tensor(_audio_array).unsqueeze(dim=0).cuda())
    _speech_embeddings = speech2embedding_model.predict(_audio_array_list, batch_size=speech_embedding_batch_size)
    return _speech_embeddings


# @default_setting([('similarity_measure' , 9, 'SIMILARITY_MEASURE',),
#                   ('speech_embedding_batch_size' , 10, 'SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE',),
#                   ('text_embedding_batch_size' , 11, 'TEXT_EMBEDDING_COMPUTATION_BATCH_SIZE',),])
# def get_parallel_data(audio_array: np.array, sample_rate: int, subtitle_list: list[pysubs2.SSAEvent],
#                        subtitle_path: Union[str, os.PathLike], subtitle_lang: Language,
#                        speech2embedding_model:SpeechToEmbeddingModelPipeline, text2embedding_model:TextToEmbeddingModelPipeline,
#                        middle_percentage:float=None, similarity_measure: Module=None,
#                        speech_embedding_batch_size:int = None, text_embedding_batch_size:int = None):

#     if middle_percentage:
#         _test_begin_index = int(len(subtitle_list)*(0.5 - (middle_percentage/2)))
#         _test_subtitle_list = subtitle_list[_test_begin_index:_test_begin_index + 50]
#         _processed_subtitles = process_subtitle_file(audio_array, sample_rate, _test_subtitle_list, subtitle_path, True)
#     else:
#         _processed_subtitles = process_subtitle_file(audio_array, sample_rate, subtitle_list, subtitle_path, False)


#     if middle_percentage:
#         _speech_embedding_batch = compute_batch_speech_embedding(audio_array, sample_rate, _processed_subtitles,
#                                                                 speech2embedding_model, speech_embedding_batch_size)
#         _processed_subtitle_content_list = []
#         for _subtitle in _processed_subtitles:
#             _processed_subtitle_content_list.append(_subtitle.content)

#         _source_language = Settings.SONAR_TEXT_LANG_MAP[subtitle_lang]
#         _text_embedding = text2embedding_model.predict(_processed_subtitle_content_list, batch_size=text_embedding_batch_size,
#                                         target_device=Settings.GPU_DEVICE, source_lang=_source_language)

#         _similarities = similarity_measure(_speech_embedding_batch, _text_embedding)
#         return _processed_subtitles, _similarities
#     return _processed_subtitles, torch.full((len(_processed_subtitles),),0.1)
