from IPython.display import Audio, display
import os, glob, zipfile, shutil, subprocess
import chardet
from typing import Optional, Union
from pathlib import Path
import requests, urllib.request
import re, copy, logging, json
from datasets import IterableDataset
import time, datetime
from IPython.display import clear_output
from tqdm.notebook import tqdm
from openai import OpenAI
import editdistance
from dataclasses import dataclass
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline

import srt
from srt import Subtitle

import datasets
import pandas as pd
import numpy as np

import ffmpeg, progressbar
import librosa
import soundfile as sf

import torch
from torch.nn import CosineSimilarity

from sonar.models.sonar_speech.loader import load_sonar_speech_model
from sonar.models.sonar_text import load_sonar_text_encoder_model , load_sonar_tokenizer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWeb'),
            ('User-agent', 'Kit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safa'),
            ('User-agent', 'ri/537.36')]
urllib.request.install_opener(opener)

class ScrapeUtilityVariableStorage:
    SAMPLING_RATE : float = 16000
    CPU_DEVICE : torch.device = torch.device("cpu")
    GPU_DEVICE : torch.device = torch.device("cuda:0")
    SIMILARITY_MEASURE = CosineSimilarity(dim=1, eps=1e-6)
    PARANTHESIS_REGEX_STR_LIST : list[str] = [r'(\[[\S\s]*\])',r'(\][\S\s]*\[)',r'(\)[\S\s]*\()',r'(\([\S\s]*\))']
    PARANTHESIS_REGEX : re.Pattern = re.compile(r'|'.join(PARANTHESIS_REGEX_STR_LIST))
    LEADING_TRAILING_WHITESPACE_REGEX : re.Pattern = re.compile(r'^\s+|\s+$')
    CONTINOUS_SYMBOL_REGEX = re.compile(r'(^(\.\.\.))|(((\.\.\.)|:)$)')
    SUBDL_SEARCH_MOVIE_URL = 'https://api.subdl.com/auto'
    SUBDL_SEARCH_SUBTITLE_URL = 'https://api.subdl.com/api/v1/subtitles'
    SUBDL_SUBTITLE_DOWNLOAD_URL = 'https://dl.subdl.com'
    MIN_LENGTH_TO_RETURN : float = 0.2
    OPENAI_API_KEY : str = None
    SUBDL_API_KEY : str = None
    SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE: int = 4
    TEXT_EMBEDDING_COMPUTATION_BATCH_SIZE: int = 8
    OPENAI_CLIENT = None
    SONAR_TEXT_LANG_MAP : dict [str, str] = {
        'fa' : 'sonar_speech_encoder_base_pes', # persian
        'en' : 'sonar_speech_encoder_base_eng', # english
        'de' : 'sonar_speech_encoder_base_deu', # german
        'fr' : 'sonar_speech_encoder_base_fra', # france
        'es' : 'sonar_speech_encoder_base_spa', # spanish
        'it' : 'sonar_speech_encoder_base_ita', # italian
        'ar' : 'sonar_speech_encoder_base_arb', # arabic
        'pt' : 'sonar_speech_encoder_base_por', # portuguese
        'ru' : 'sonar_speech_encoder_base_rus', # russian
        'tr' : 'sonar_speech_encoder_base_tur', # turkish
        'uk' : 'sonar_speech_encoder_base_ukr', # ukrainian
        'pl' : 'sonar_speech_encoder_base_pol', # polish
        'zh' : 'sonar_speech_encoder_base_cmn', # chinese
        'hi' : 'sonar_speech_encoder_base_hin', # hindi
        'bn' : 'sonar_speech_encoder_base_ben', # bengali
        'ja' : 'sonar_speech_encoder_base_jpn', # japanese
        'vi' : 'sonar_speech_encoder_base_vie', # vietnamese
        'ko' : 'sonar_speech_encoder_base_kor', # korean
        'fi' : 'sonar_speech_encoder_base_fin', # finnish
        'ur' : 'sonar_speech_encoder_base_urd'  # Urdu
    }

    SONAR_SPEECH_ENCODER_LANG_MAP : dict [str, str] = {
        'fa' : 'pes_Arab',      # persian
        'en' : 'eng_Latn',      # english
        'de' : 'deu_Latn',      # german
        'fr' : 'fra_Latn',      # france
        'es' : 'spa_Latn',      # spanish
        'it' : 'ita_Latn',      # italian
        'ar' : 'arb_Arab',      # arabic
        'pt' : 'por_Latn',      # portuguese
        'ru' : 'rus_Cyrl',      # russian
        'tr' : 'tur_Latn',      # turkish
        'uk' : 'ukr_Cyrl',      # ukrainian
        'pl' : 'pol_Latn',      # polish
        'zh' : 'zho_Hans',      # chinese
        'hi' : 'hin_Deva',      # hindi
        'bn' : 'ben_Beng',      # bengali
        'ja' : 'jpn_Jpan',      # japanese
        'vi' : 'vie_Latn',      # vietnamese
        'ko' : 'kor_Hang',      # korean
        'fi' : 'fin_Latn',      # finnish
        'ur' : 'urd_Arab'       # Urdu
    }

    SUBDL_LANG_CODES : dict [str, list[str]] = {
        'fa' : ['FA'],          # persian
        'en' : ['EN'],          # english
        'de' : ['DE', 'EN_DE'], # german
        'fr' : ['FR'],          # france
        'es' : ['ES'],          # spanish
        'it' : ['IT'],          # italian
        'ar' : ['AR'],          # arabic
        'pt' : ['PT'],          # portuguese
        'ru' : ['RU'],          # russian
        'tr' : ['TR'],          # turkish
        'uk' : ['UK'],          # ukrainian
        'pl' : ['PL'],          # polish
        'zh' : ['ZH_BG', 'ZH'], # chinese
        'hi' : ['HI'],          # hindi
        'bn' : ['BN'],          # bengali
        'ja' : ['JA'],          # japanese
        'vi' : ['VI'],          # vietnamese
        'ko' : ['KO'],          # korean
        'fi' : ['FI'],          # finnish
        'ur' : ['UR']           # Urdu
    }

    SUBDL_LANGUAGE_MAP : dict[str, str] = {
    "farsi_persian" : 'fa',     # persian
    "english" : 'en',           # english
    "english_german" : 'de',    # german
    "german" : 'de',            # german
    "french" : 'fr',            # france
    "spanish" : 'es',           # spanish
    "italian" : 'it',           # italian
    "arabic" : 'ar',            # arabic
    "portuguese" : 'pt',        # portuguese
    "russian" : 'ru',           # russian
    "turkish" : 'tr',           # turkish
    "ukranian" : 'uk',          # ukrainian
    "polish": 'pl',             # polish
    "big 5 code": 'zh',         # chinese
    "chinese bg code" : 'zh',   # chinese
    "hindi" : "hi",             # hindi
    "bengali" : "bn",           # bengali
    "japanese" : "ja",          # japanese
    "vietnamese" : "vi",        # vietnamese
    "korean" : "ko",            # korean
    "finnish" : "fi",           # finnish
    "urdu" : "ur"               # Urdu
    }
    


def save_json(path: Union[str, os.PathLike], obj):
    with open(path,  'w',  encoding='utf-8') as f:
        json.dump(obj,  f,  ensure_ascii=False)

def load_json(path: Union[str, os.PathLike]):
    with open(path, 'r') as f:
        _obj = json.load(f)
    return _obj

class ShowProgressUrllib:
    def __init__(self, description: str):
        self._pre_block_num = -1
        self._description = description
        self._tqdm_pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self._tqdm_pbar is None:
            self._tqdm_pbar = tqdm(desc=self._description,total=(total_size), unit='B', unit_scale=True)

        _all_downloaded_bytes = (block_num * block_size)
        _current_downloaded_bytes = ((block_num - self._pre_block_num) * block_size)
        self._pre_block_num = block_num
        if _all_downloaded_bytes < total_size:
            self._tqdm_pbar.update(_current_downloaded_bytes)
        else:
            self._tqdm_pbar.clear()
            self._tqdm_pbar.close()
            self._pre_block_num = -1

def create_directory(directory: Union[str, os.PathLike]):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_segment_from_array(audio_array: np.array, sample_rate: int, subtitle: Subtitle, min_length_to_return: float = 0.2):
    _segment_start_index = int(subtitle.start.total_seconds()*sample_rate)
    _segment_end_index = int(subtitle.end.total_seconds()*sample_rate)
    _segment_array = audio_array[_segment_start_index: _segment_end_index]
    _min_length = int(min_length_to_return * sample_rate)
    if _segment_array.shape[0] > _min_length:
        return _segment_array
    else:
        return np.zeros(shape=(_min_length,), dtype=np.float32)

def show_parallel_data(audio_array: np.array, sample_rate: int, subtitle):
    print(f'subtitle : {subtitle.content}')
    _audio_array = extract_segment_from_array(audio_array, sample_rate, subtitle)
    display(Audio(_audio_array, rate=sample_rate))

def compute_similarity(audio_array: np.array, sample_rate: int, subtitle: Subtitle, speech_embedding_model, text_embedding_model):
    _audio_array = extract_segment_from_array(audio_array, sample_rate, subtitle)
    _speech_embedding = speech_embedding_model.predict([torch.Tensor(_audio_array).unsqueeze(dim=0)])
    _text_embedding = text_embedding_model.predict([subtitle.content], source_lang='pes_Arab')
    return ScrapeUtilityVariableStorage.SIMILARITY_MEASURE(_speech_embedding, _text_embedding)

def compute_batch_speech_embedding(audio_array: np.array, sample_rate: float, subtitle_list: list[Subtitle], speech2embedding_model:SpeechToEmbeddingModelPipeline):
    _audio_array_list = []
    for _subtitle in subtitle_list:
        _audio_array = extract_segment_from_array(audio_array, sample_rate, _subtitle)
        _audio_array_list.append(torch.Tensor(_audio_array).unsqueeze(dim=0).cuda())
    _speech_embeddings = speech2embedding_model.predict(_audio_array_list, batch_size=ScrapeUtilityVariableStorage.SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE)
    return _speech_embeddings

def get_min_download_bar(movie, encoder : str = None):
    _minsize : int = 999_999_999
    _minsize_item = None
    _minsize_index : int = 0
    for _minsize_idx, _download_bar in enumerate(movie['download_results'][0]['download_bars']):
        if _download_bar['size'] < _minsize:
            if encoder:
                if not (_download_bar['encoder'].lower() == encoder.lower()):
                    continue
            _minsize = _download_bar['size']
            _minsize_item = _download_bar
            _minsize_index = _minsize_idx
    return _minsize_item, _minsize_index

def get_best_download_bar(movie):
    _minsize_item, _minsize_index = get_min_download_bar(movie)
    _minsize_item_encoder_30nama, _minsize_index_encoder_30nama = get_min_download_bar(movie, '30nama')
    if _minsize_item_encoder_30nama:
        return _minsize_index_encoder_30nama, _minsize_item_encoder_30nama
    return _minsize_index, _minsize_item

def find_movies_subdl(title: str, year: int):
    _payload = {'query' : title}
    _response = requests.get(ScrapeUtilityVariableStorage.SUBDL_SEARCH_MOVIE_URL, params=_payload).json()
    for _posible_movie in _response['results']:
        _posible_movie['score'] = editdistance.eval(title, _posible_movie['name'])
        if _posible_movie['year'] == year:
            _posible_movie['score'] = _posible_movie['score'] - 5
    _response['results'].sort(key=lambda x: x['score'])
    return _response['results']

class SubdlException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)
 
class ScarapingException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)

class CNamaDownloadException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)

def find_subtitle(title: str, year: int, languages: list[str]):
    _languages_subdl = ','.join(languages)
    _payload = {'api_key': ScrapeUtilityVariableStorage.SUBDL_API_KEY, 'film_name' : title, 'languages' : _languages_subdl}
    if type(year) == int and year >= 1800 and year <= 2024:
        _payload['year'] = year
    try:
        _response = requests.get(ScrapeUtilityVariableStorage.SUBDL_SEARCH_SUBTITLE_URL, params=_payload).json()
    except Exception:
        raise SubdlException(f'Subdl api can is not working.')
    if _response.status_code != 200:
        raise SubdlException(f'Subdl api returned status code {requests.status_codes}')
    if _response['status'] == False:
        raise SubdlException(f'Subdl result status was false')
    else:
        return _response['subtitles']

def extract_title_year_from_30nama_title(movie):
    if len(movie['title']) < 6:
        raise ScarapingException('movie title is too short. we can not extract title and year from it.')
    _title_subdl = movie['title'][:-5]
    try:
        _year_subdl = int(movie['title'][-4:])
    except ValueError:
        raise ScarapingException('year can not be converted to integer.')
    return _title_subdl, _year_subdl

def download_subtitle_subdl(movie, languages: list[str], processed_langs, max_subtitle_movie_try: int):
    _movie_id = movie['id']
    _title_subdl, _year_subdl = extract_title_year_from_30nama_title(movie)
    _posible_movies = find_movies_subdl(_title_subdl, _year_subdl, languages)
    for idx_posible_movie_id, _posible_movie in enumerate(_posible_movies[:max_subtitle_movie_try]):
        _subdl_subtitles = find_subtitle(_posible_movie['name'] ,_posible_movie['year'])
        for _subtitle_id, _subtitle in enumerate(_subdl_subtitles):
            if _subtitle['lang'].lower() in ScrapeUtilityVariableStorage.SUBDL_LANGUAGE_MAP:
                _subtitle_lang = ScrapeUtilityVariableStorage.SUBDL_LANGUAGE_MAP[_subtitle['lang'].lower()]
            else:
                logging.warn(f"this language({_subtitle['lang'].lower()}) is  not supported in our snippet but was returned by the api.")
                continue
            if _subtitle_lang in processed_langs:
                continue
            if (_subtitle_lang not in languages):
                logging.warn(f"this language ({_subtitle['lang'].lower()}) is not wanted but is in the subdl api results.\npossible movie: {_posible_movie['name']}\nspossible movie year{_posible_movie['year']}\n returned subtitle object : {_subtitle}\n")
                continue
            create_directory(f'./temp/{_movie_id}/')
            if not _subtitle['url']:
                raise Exception(f"empty subtitle url {_subtitle['url']} is not wanted but is in the subdl api results.")
            _zip_path = f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{_subtitle_lang}.zip'
            if os.path.isfile(_zip_path):
                continue
            _zipped_url = ScrapeUtilityVariableStorage.SUBDL_SUBTITLE_DOWNLOAD_URL + '/' + _subtitle['url']
            try:
                urllib.request.urlretrieve(_zipped_url,_zip_path)
            except Exception:
                raise SubdlException(f'can not retrive following subd subtitle :{_zipped_url}')
            with zipfile.ZipFile(_zip_path, 'r') as _zip_ref:
                _zip_ref.extractall(f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{_subtitle_lang}/')
                _srt_list = glob.glob(f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{_subtitle_lang}/?*.srt')
                for _subtitle_path in _srt_list:
                    yield _subtitle_path, _subtitle_lang

def download_video_file(download_bar, download_bar_index, movie, sampling_rate : float):
    movie_id = movie['id']
    if not os.path.isfile(f'./dataset/{movie_id}/all.wav'):
        _download_link = download_bar['download_link']
        _movie_extention = _download_link.split('/')[-1].split('.')[-1]
        _video_file_path = f'./temp/{movie_id}/{download_bar_index}.{_movie_extention}'
        try:
            urllib.request.urlretrieve(_download_link, _video_file_path, ShowProgressUrllib())
        except Exception:
            CNamaDownloadException('Today maximum requests have reached its limits.')
        ffmpeg.input(_video_file_path).output(f'./dataset/{movie_id}/all.mp3', ac=1, ar=sampling_rate).run()
        return True

def get_parallel_data(audio_array: np.array, sample_rate: int, subtitle_list: list[Subtitle], subtitle_path: Union[str, os.PathLike], subtitle_lang: str, speech2embedding_model:SpeechToEmbeddingModelPipeline, text2embedding_model:TextToEmbeddingModelPipeline, middle_percentage:float=None):

    if middle_percentage:
        _test_begin_index = int(len(subtitle_list)*(0.5 - (middle_percentage/2)))
        _test_subtitle_list = subtitle_list[_test_begin_index:_test_begin_index + 50]
        _processed_subtitles = process_subtitle_file(audio_array, sample_rate, _test_subtitle_list, subtitle_path, True)
    else:
        _processed_subtitles = process_subtitle_file(audio_array, sample_rate, subtitle_list, subtitle_path, False)


    if middle_percentage:
        _speech_embedding_batch = compute_batch_speech_embedding(audio_array, sample_rate, _processed_subtitles,
                                                                speech2embedding_model, ScrapeUtilityVariableStorage.SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE)
        _processed_subtitle_content_list = []
        for _subtitle in _processed_subtitles:
            _processed_subtitle_content_list.append(_subtitle.content)

        _source_language = ScrapeUtilityVariableStorage.SONAR_TEXT_LANG_MAP[subtitle_lang]
        _text_embedding = text2embedding_model.predict(_processed_subtitle_content_list, batch_size=ScrapeUtilityVariableStorage.TEXT_EMBEDDING_COMPUTATION_BATCH_SIZE,
                                        target_device=ScrapeUtilityVariableStorage.GPU_DEVICE, source_lang=_source_language)

        _similarities = ScrapeUtilityVariableStorage.SIMILARITY_MEASURE(_speech_embedding_batch, _text_embedding)
        return _processed_subtitles, _similarities
    return _processed_subtitles, torch.full((len(_processed_subtitles),),0.1)

def legacy_algo(segments_list):
    SUBTITLE_MAX_GAP = 2 # in seconds
    _is_contimous_list = []
    _index_to_be_computed = []
    _final_output_dict = {}
    _prompt_list = []
    for _idx, (_segment1, _segment2) in enumerate(zip(segments_list[:-1],segments_list[1:])):
        if (_segment2.start.total_seconds() > _segment1.end.total_seconds() + SUBTITLE_MAX_GAP):
            _final_output_dict[_idx] = False
        elif preprocess_subtitle_str(_segment1.content) and preprocess_subtitle_str(_segment2.content):
            instruction = 'Respond with either yes or no if a meaningfull bigger sentence is split between these two subtitle blocks.'
            question_prompt = 'Subtitle block : \n{}\nSubtitle block : \n{}\n\n{}'
            _a = '[' + str(_segment1.start).split(".")[0] + ' --> ' + str(_segment1.end).split('.')[0] + ']\n' + _segment1.content
            _b = '[' + str(_segment2.start).split(".")[0] + ' --> ' + str(_segment2.end).split('.')[0] + ']\n' + _segment2.content
            _prompt_str = f'{instruction}\n\n{question_prompt.format(_a,_b,"")}'
            _prompt_list.append(_prompt_str)
            _index_to_be_computed.append(_idx)
        else:
            _final_output_dict[_idx] = True
    _choices = []
    for i in range(0,len(_prompt_list),20):
        _choices.extend(ScrapeUtilityVariableStorage.OPENAI_CLIENT.completions.create(model="gpt-3.5-turbo-instruct",
                                              prompt=_prompt_list[i:i+20]
                                              , max_tokens=3, temperature=0.1).choices)
    for id,choice in enumerate(_choices):
        _index_prompt = _index_to_be_computed[id]
        _response = choice.text.replace('\n', '')
        _response = ScrapeUtilityVariableStorage.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _response)
        if _response.lower() == 'yes':
            _final_output_dict[_index_prompt] = True
        else:
            _final_output_dict[_index_prompt] = False
    for i in range(len(segments_list)-1):
        _is_contimous_list.append(_final_output_dict[i])
    return _is_contimous_list

def load_subtitle_file(subtitle_path):
    _subtitle_encoding  = predict_encoding(subtitle_path)
    try:
        _file = open(subtitle_path, encoding=_subtitle_encoding)
        _file_content = _file.read()
        _srt_parsed_generator = srt.parse(_file_content)
        return list(_srt_parsed_generator)
    except Exception as e:
        logging.warn(f'subtitle can not be procdessed.\nsubtitle_path: {subtitle_path}\nsubtitle predicted encoding: {_subtitle_encoding}')
        return []

def sybc_and_load_subtitle_file(subtitle_path, movie):
    _movie_id = movie['id']
    _audio_path = f'./dataset/{_movie_id}/all.mp3'
    _sync_path = '/'.join(subtitle_path.split('/')[:-1]) + '/sync.srt'
    subprocess.run(["ffs", _audio_path, '-i', subtitle_path, '-o', _sync_path])
    if os.path.exists(_sync_path):
        return load_subtitle_file(_sync_path), _sync_path
    else:
        logging.warn(f'ffs can not sync the subtitle.\nsubtitle_path: {subtitle_path}')
        return [] , ''

def preprocess_subtitle_str(subtitle_str):
    _subtitle_str = subtitle_str.replace('\u200c',' ')
    _subtitle_str = _subtitle_str.replace('\n',' ')
    _subtitle_str = ScrapeUtilityVariableStorage.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.PARANTHESIS_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.CONTINOUS_SYMBOL_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _subtitle_str)
    return _subtitle_str

def zip_wavs(movie):
    _movie_id = movie['id']
    with zipfile.ZipFile(f'./dataset/{_movie_id}/{_movie_id}.zip', 'w', zipfile.ZIP_DEFLATED) as _zipf:
        _wavs = glob.glob(f'./dataset/{_movie_id}/*.mp3')
        for _wav in _wavs:
            _zipf.write(_wav, _wav.split('/')[-1], compress_type=zipfile.ZIP_DEFLATED)
            os.remove(_wav)

def save_results(movie, audio_array: np.array, sample_rate: float, movie_in_minutes, movie_in_seconds, final_subtitles: list[Subtitle], subtitle_lang: str, similarities: np.array):
    pd_dict = {}
    movie_id = movie['id']
    pd_dict['start'] = [subtitle.start.total_seconds() for subtitle in final_subtitles]
    pd_dict['end'] = [subtitle.end.total_seconds() for subtitle in final_subtitles]
    pd_dict['content'] = [subtitle.content for subtitle in final_subtitles]
    pd_dict['sub_index'] = [subtitle.index for subtitle in final_subtitles]
    pd_dict['lang'] = [subtitle_lang]*len(final_subtitles)
    pd_dict['movie_id'] = [movie_id]*len(final_subtitles)
    pd_dict['movie_duration_in_minutes'] = [movie_in_minutes]*len(final_subtitles)
    pd_dict['movie_duration_in_seconds'] = [movie_in_seconds]*len(final_subtitles)
    pd_dict['segment_duration'] = [(subtitle.end - subtitle.start).total_seconds() for subtitle in final_subtitles]
    pd_dict['score'] = similarities.detach().cpu().numpy()
    df = pd.DataFrame.from_dict(pd_dict, 'columns')
    df = df.sort_values('score', ascending=False).drop_duplicates(['movie_id','sub_index'],keep='first')
    segment_path_list = []
    for index, row in df.iterrows():
        segment_array = audio_array[int(row['start']*sample_rate): int(row['end']*sample_rate)]
        sf.write(f'./dataset/{movie_id}/{row["sub_index"]}.mp3', segment_array, sample_rate, format='wav')
        segment_path_list.append(f'{movie_id}/{row["sub_index"]}.mp3')
    df['path'] = segment_path_list
    df.to_csv(f'./dataset/{subtitle_lang}_{movie_id}.csv', index=False)
    zip_wavs(movie_id)


def get_movie_str(movie):
    return f'movie title: {movie["title"]} ### movie id: {movie["idgr"]}'

def check_movie(movie):
    if movie['is_series']:
        return False
    if 'subtitle_results' not in movie:
        return False
    if 'download_results' not in movie:
        return False
    if len(movie['download_results']) == 0:
        logging.info(f"{get_movie_str(movie)}\nbecause movie has 0 download_results it was dropped.")
        return False
    if len(movie['download_results']) < 1:
        logging.info(f"{get_movie_str(movie)}\nbeacause movie has more than 1 download_results it was dropped.")
        return False
    if len(movie['download_results'][0]) == 0:
        logging.info(f"{get_movie_str(movie)}\nbeacause movie has 0 download bars it was dropped.")
        return False
    if len(movie['subtitle_results']) == 0:
        logging.info(f"{get_movie_str(movie)}\nbeacause movie has 0 subtitle it was dropped.")
        return False
    return True

def predict_encoding(file_path: Path, n_lines: int=20) -> str:
    '''Predict a file's encoding using chardet'''

    if file_path.find('[ANSI]') != -1:
        return 'cp1256'

    # Open the file as binary data
    with Path(file_path).open('rb') as f:
        # Join binary lines for specified number of lines
        _rawdata = b''.join([f.readline() for _ in range(n_lines)])

    return chardet.detect(_rawdata)['encoding']

def divide_subtitle_list(subtitle_list, continous_list):
    _subtitle_division_list = []
    _division = []
    assert len(continous_list) + 1 == len(subtitle_list)
    for _subtitle, _is_continous in zip(subtitle_list, [True] + continous_list):
        if _division and (not _is_continous):
            _subtitle_division_list.append(_division)
            _division = []
        _division.append(_subtitle)    
    if _division:
        _subtitle_division_list.append(_division)
    return _subtitle_division_list

def process_subtitle_file(subtitle_list):
    _processed_subtitles = []
    continous_truth_list = legacy_algo(subtitle_list)
    _subtitle_division_list = divide_subtitle_list(subtitle_list,continous_truth_list)
    for _division_idx, _division in enumerate(_subtitle_division_list):
        _start = _division[0].start
        _end = _division[-1].end
        _division_content = ''
        for _subtitle in _division:
            _division_content += preprocess_subtitle_str(_subtitle.content)
        _division_content = preprocess_subtitle_str(_division_content)
        _processed_subtitles.append(Subtitle(_division_idx + 1, _start, _end, _division_content))
    return _processed_subtitles

# nlp = Pipeline(lang="multilingual", processors="langid")
# docs = ["Hello world.", "Bonjour le monde!"]
# docs = [Document([], text=text) for text in docs]
# nlp(docs)
# print("\n".join(f"{doc.text}\t{doc.lang}" for doc in docs)) 