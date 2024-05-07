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


class ScrapeUtilityVariableStorage:
    CPU_DEVICE : torch.device = torch.device("cpu")
    GPU_DEVICE : torch.device = torch.device("cuda:0")
    SIMILARITY_MEASURE = CosineSimilarity(dim=1, eps=1e-6)
    PARANTHESIS_REGEX_STR_LIST : list[str] = [r'(\[[\S\s]*\])',r'(\][\S\s]*\[)',r'(\)[\S\s]*\()',r'(\([\S\s]*\))']
    PARANTHESIS_REGEX : re.Pattern = re.compile(r'|'.join(PARANTHESIS_REGEX_STR_LIST))
    WHITESPACE_REGEX : re.Pattern = re.compile(r'^\s+|\s+$')
    CONTINOUS_SYMBOL_REGEX = re.compile(r'(^(\.\.\.))|(((\.\.\.)|:)$)')
    SUBDL_SEARCH_MOVIE_URL = 'https://api.subdl.com/auto'
    SUBDL_SEARCH_SUBTITLE_URL = 'https://api.subdl.com/api/v1/subtitles'
    SUBDL_SUBTITLE_DOWNLOAD_URL = 'https://dl.subdl.com'
    MIN_LENGTH_TO_RETURN : float = 0.2
    OPENAI_API_KEY : str = None
    SUBDL_API_KEY : str = None
    SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE: int = 4
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
        'ur' : 'sonar_speech_encoder_base_urd' # Urdu
    }

    SONAR_SPEECH_ENCODER_LANG_MAP : dict [str, str] = {
        'fa' : 'pes_Arab', # persian
        'en' : 'eng_Latn', # english
        'de' : 'deu_Latn', # german
        'fr' : 'fra_Latn', # france
        'es' : 'spa_Latn', # spanish
        'it' : 'ita_Latn', # italian
        'ar' : 'arb_Arab', # arabic
        'pt' : 'por_Latn', # portuguese
        'ru' : 'rus_Cyrl', # russian
        'tr' : 'tur_Latn', # turkish
        'uk' : 'ukr_Cyrl', # ukrainian
        'pl' : 'pol_Latn', # polish
        'zh' : 'zho_Hans', # chinese
        'hi' : 'hin_Deva', # hindi
        'bn' : 'ben_Beng', # bengali
        'ja' : 'jpn_Jpan', # japanese
        'vi' : 'vie_Latn', # vietnamese
        'ko' : 'kor_Hang', # korean
        'fi' : 'fin_Latn', # finnish
        'ga' : 'gle_Latn', # Irish
        'ur' : 'urd_Arab' # Urdu
    }

    SUBDL_LANGUAGE_MAP : dict[str, str] = {
    "AR": "Arabic",
    "BR_PT": "Brazillian Portuguese",
    "DA": "Danish",
    "NL": "Dutch",
    "EN": "English",
    "FA": "Farsi_Persian",
    "FI": "Finnish",
    "FR": "French",
    "ID": "Indonesian",
    "IT": "Italian",
    "NO": "Norwegian",
    "RO": "Romanian",
    "ES": "Spanish",
    "SV": "Swedish",
    "VI": "Vietnamese",
    "SQ": "Albanian",
    "AZ": "Azerbaijani",
    "BE": "Belarusian",
    "BN": "Bengali",
    "ZH_BG": "Big 5 code",
    "BS": "Bosnian",
    "BG": "Bulgarian",
    "BG_EN": "Bulgarian_English",
    "MY": "Burmese",
    "CA": "Catalan",
    "ZH": "Chinese BG code",
    "HR": "Croatian",
    "CS": "Czech",
    "NL_EN": "Dutch_English",
    "EN_DE": "English_German",
    "EO": "Esperanto",
    "ET": "Estonian",
    "KA": "Georgian",
    "DE": "German",
    "EL": "Greek",
    "KL": "Greenlandic",
    "HE": "Hebrew",
    "HI": "Hindi",
    "HU": "Hungarian",
    "HU_EN": "Hungarian_English",
    "IS": "Icelandic",
    "JA": "Japanese",
    "KO": "Korean",
    "KU": "Kurdish",
    "LV": "Latvian",
    "LT": "Lithuanian",
    "MK": "Macedonian",
    "MS": "Malay",
    "ML": "Malayalam",
    "MNI": "Manipuri",
    "PL": "Polish",
    "PT": "Portuguese",
    "RU": "Russian",
    "SR": "Serbian",
    "SI": "Sinhala",
    "SK": "Slovak",
    "SL": "Slovenian",
    "TL": "Tagalog",
    "TA": "Tamil",
    "TE": "Telugu",
    "TH": "Thai",
    "TR": "Turkish",
    "UK": "Ukranian",
    "UR": "Urdu"
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

def compute_batch_speech_embedding(audio_array: np.array, sample_rate: float, subtitle_list: list[Subtitle], speech_embedding_model:SpeechToEmbeddingModelPipeline, speech_embedding_batch_size: int):
    _audio_array_list = []
    for subtitle in subtitle_list:
        _audio_array = extract_segment_from_array(audio_array, sample_rate, subtitle)
        _audio_array_list.append(torch.Tensor(_audio_array).unsqueeze(dim=0).cuda())
    _speech_embeddings = speech_embedding_model.predict(_audio_array_list, batch_size=speech_embedding_batch_size)
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

def find_subtitle(title: str, year: int, languages: list[str], subdl_api_key: str):
    _languages_subdl = ','.join(languages)
    _payload = {'api_key': subdl_api_key, 'film_name' : title, 'languages' : _languages_subdl}
    if type(year) == int and year >= 1800 and year <= 2024:
        _payload['year'] = year
    _response = requests.get(ScrapeUtilityVariableStorage.SUBDL_SEARCH_SUBTITLE_URL, params=_payload).json()
    if _response['status'] == False:
        return []
    else:
        return _response['subtitles']

def download_subtitle_subdl(movie, languages: list[str], processed_langs, max_subtitle_movie_try: int):
    _movie_id = get_movie_id(movie)
    _title_subdl = movie['title'][:-5]
    _year_subdl = int(movie['title'][-4:])
    _posible_movies = find_movies_subdl(_title_subdl, _year_subdl, languages)
    for idx_posible_movie_id, _posible_movie in enumerate(_posible_movies[:max_subtitle_movie_try]):
        _subdl_subtitles = find_subtitle(_posible_movie['name'] ,_posible_movie['year'])
        for _subtitle_id, _subtitle in enumerate(_subdl_subtitles):
            if _subtitle['lang'] == 'farsi_persian':
                subtitle_link_lang = 'fa'
            elif _subtitle['lang'] == 'english':
                subtitle_link_lang = 'en'
            else:
                raise Exception(f"this language ({_subtitle['lang']}) is  not supported.")
            try:
                if subtitle_link_lang in processed_langs or (subtitle_link_lang not in languages):
                    break
                if _subtitle['url']:
                    create_directory(f'./temp/{_movie_id}/')
                    zip_path = f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{subtitle_link_lang}.zip'
                    if not os.path.isfile(zip_path):
                        zipped_url = ScrapeUtilityVariableStorage.SUBDL_SUBTITLE_DOWNLOAD_URL + '/' + _subtitle['url']
                        opener = urllib.request.build_opener()
                        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWeb'),
                                    ('User-agent', 'Kit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safa'),
                                    ('User-agent', 'ri/537.36')]
                        urllib.request.install_opener(opener)
                        urllib.request.urlretrieve(zipped_url,zip_path)
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{subtitle_link_lang}/')
                    srt_list = glob.glob(f'./temp/{_movie_id}/sub_{idx_posible_movie_id}_{_subtitle_id}_{subtitle_link_lang}/?*.srt')
                    if len(srt_list) > 0:
                        yield srt_list[0], subtitle_link_lang
                    subtitle_exist = True
            except Exception as e:
                continue

def get_movie_id(movie):
    return movie["page_num"] * 100 + movie["index"]


def download_video_file(download_bar, download_bar_index, movie, sampling_rate : float):
    movie_id = get_movie_id(movie)
    if not os.path.isfile(f'./dataset/{movie_id}/all.wav'):
        _download_link = download_bar['download_link']
        _movie_extention = _download_link.split('/')[-1].split('.')[-1]
        _video_file_path = f'./temp/{movie_id}/{download_bar_index}.{_movie_extention}'
        urllib.request.urlretrieve(_download_link, _video_file_path, ShowProgressUrllib())
        ffmpeg.input(_video_file_path).output(f'./dataset/{movie_id}/all.wav', ac=1, ar=sampling_rate).run()
        return True

# def load_speech_model_in_gpu():
#     speech_encoder_model.cuda()
#     s2vec_model.cuda()
#     text_encoder_model.cpu()
#     t2vec_model.device = device_cpu
#     t2vec_model.cpu()

# def load_text_model_in_gpu():
#     speech_encoder_model.cpu()
#     s2vec_model.cpu()
#     text_encoder_model.cuda()
#     t2vec_model.device = device_gpu
#     t2vec_model.cuda()

def get_parallel_data(audio_array: np.array, sample_rate: int, subtitle_list: list[Subtitle], subtitle_path: Union[str, os.PathLike], subtitle_lang: str, middle_percentage:float=None):
    # load_speech_model_in_gpu()

    if middle_percentage:
        test_begin_index = int(len(subtitle_list)*(0.5 - (middle_percentage/2)))
        test_subtitle_list = subtitle_list[test_begin_index:test_begin_index + 50]
        processed_subtitles = process_subtitle_file(audio_array, sample_rate, test_subtitle_list, subtitle_path, True)
    else:
        processed_subtitles = process_subtitle_file(audio_array, sample_rate, subtitle_list, subtitle_path, False)


    if middle_percentage:
        speech_embedding_batch = compute_batch_speech_embedding(audio_array, sample_rate, processed_subtitles,
                                                                s2vec_model, SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE)
        processed_subtitle_content_list = []
        for subtitle in processed_subtitles:
            processed_subtitle_content_list.append(subtitle.content)

        if subtitle_lang == 'fa':
            text_embedding = t2vec_model.predict(processed_subtitle_content_list, batch_size=TEXT_EMBEDDING_BATCH_SIZE,
                                        target_device=GPU_DEVICE, source_lang='pes_Arab')
        if subtitle_lang == 'en':
            text_embedding = t2vec_model.predict(processed_subtitle_content_list, batch_size=TEXT_EMBEDDING_BATCH_SIZE,
                                        target_device=GPU_DEVICE, source_lang='eng_Latn')

        similarities = similarity_meausre(speech_embedding_batch, text_embedding)
        return processed_subtitles, similarities
    return processed_subtitles, torch.full((len(processed_subtitles),),0.1)

def is_segments_continous_batch(segments_list):
    _is_contimous_list = []
    _index_to_be_computed = []
    _final_output_dict = {}
    _prompt_list = []
    for _idx, (_segment1, _segment2) in enumerate(zip(segments_list[:-1],segments_list[1:])):
        if (_segment2.start.total_seconds() > _segment1.end.total_seconds() + 2):
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
        _choices.extend(client.completions.create(model="gpt-3.5-turbo-instruct",
                                              prompt=_prompt_list[i:i+20]
                                              , max_tokens=3, temperature=0.1).choices)
    for id,choice in enumerate(_choices):
        _index_prompt = _index_to_be_computed[id]
        _response = choice.text.replace('\n', '')
        _response = ScrapeUtilityVariableStorage.WHITESPACE_REGEX.sub('', _response)
        if _response.lower() == 'yes':
            _final_output_dict[_index_prompt] = True
        else:
            _final_output_dict[_index_prompt] = False
    for i in range(len(segments_list)-1):
        _is_contimous_list.append(_final_output_dict[i])
    return _is_contimous_list

def load_subtitle_file(subtitle_path):
    subtitle_encoding  = predict_encoding(subtitle_path)
    try:
        _file = open(subtitle_path, encoding=subtitle_encoding)
        _file_content = _file.read()
        _srt_parsed_generator = srt.parse(_file_content)
        return list(_srt_parsed_generator)
    except Exception as e:
        return []

def sybc_and_load_subtitle_file(subtitle_path, movie):
    _movie_id = get_movie_id(movie)
    _audio_path = f'./dataset/{_movie_id}/all.wav'
    _sync_path = '/'.join(subtitle_path.split('/')[:-1]) + '/sync.srt'
    subprocess.run(["ffs", _audio_path, '-i', subtitle_path, '-o', _sync_path])
    if os.path.exists(_sync_path):
        return load_subtitle_file(_sync_path), _sync_path
    return [] , ''

def preprocess_subtitle_str(subtitle_str):
    _subtitle_str = subtitle_str.replace('\u200c','')
    _subtitle_str = _subtitle_str.replace('\n',' ')
    _subtitle_str = _subtitle_str.replace('...','')
    _subtitle_str = ScrapeUtilityVariableStorage.WHITESPACE_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.PARANTHESIS_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.CONTINOUS_SYMBOL_REGEX.sub('', _subtitle_str)
    _subtitle_str = ScrapeUtilityVariableStorage.WHITESPACE_REGEX.sub('', _subtitle_str)
    return _subtitle_str

def zip_wavs(movie):
    _movie_id = get_movie_id(movie)
    with zipfile.ZipFile(f'./dataset/{_movie_id}/{_movie_id}.zip', 'w', zipfile.ZIP_DEFLATED) as _zipf:
        _wavs = glob.glob(f'./dataset/{_movie_id}/*.wav')
        for _wav in _wavs:
            _zipf.write(_wav, _wav.split('/')[-1], compress_type=zipfile.ZIP_DEFLATED)
            os.remove(_wav)

def save_results(movie, audio_array: np.array, sample_rate: float, movie_in_minutes, movie_in_seconds, final_subtitles: list[Subtitle], subtitle_lang: str, similarities: np.array):
    pd_dict = {}
    movie_id = get_movie_id(movie)
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
        sf.write(f'./dataset/{movie_id}/{row["sub_index"]}.wav', segment_array, sample_rate, format='wav')
        segment_path_list.append(f'{movie_id}/{row["sub_index"]}.wav')
    df['path'] = segment_path_list
    df.to_csv(f'./dataset/{subtitle_lang}_{movie_id}.csv', index=False)
    zip_wavs(movie_id)

def check_movie(movie):
    if movie['is_series']:
        return False
    if 'subtitle_results' not in movie:
        return False
    if 'download_results' not in movie:
        return False
    if len(movie['download_results']) != 1:
        logging.info(f"for movie {movie['title']} len(movie['download_results']) != 1: --x-- " + str(movie))
        return False
    if len(movie['download_results'][0]) == 0:
        logging.info(f"for movie {movie['title']} len(movie['download_results'][0]) == 0 --x-- " + str(movie))
        return False
    if len(movie['subtitle_results']) == 0:
        logging.info(f"for movie {movie['title']} len(movie['subtitle_results']) == 0 --x-- " + str(movie))
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

def process_subtitle_file(subtitle_list):
    _start = None
    _end = None
    _content = ''
    _processed_subtitles = []
    _count = 0
    _time_gap = datetime.timedelta(seconds=0.2)
    _num_continous = 0
    _is_segments_continous_list = is_segments_continous_batch(subtitle_list)
    for _subt_idx, _subtitle in enumerate(tqdm(subtitle_list,desc='chatgpt')):
        # time.sleep(0.17)
        if _subt_idx == 0:
            is_continous = True
        else:
            is_continous = _is_segments_continous_list[_subt_idx-1]
        _content += ' ' + _subtitle.content
        _content = preprocess_subtitle_str(_content)
        _num_continous += 1
        _end = _subtitle.end
        if not _start:
            _start = _subtitle.start
        if (not is_continous) or (_num_continous > 3) or (_subt_idx == (len(subtitle_list) - 1)):
            if _content and ((_end.total_seconds() - _start.total_seconds()) < 30):
                _count+=1
                processed_subtitle = Subtitle(_count, _start, _end, _content)
                _processed_subtitles.append(processed_subtitle)
            _num_continous = 0
            _start = None
            _end = None
            _content = ''
    return _processed_subtitles