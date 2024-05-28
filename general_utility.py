from typing import  Union
from pathlib import Path
import logging, json
import os, glob, zipfile
from IPython.display import Audio, display    
import chardet
from typing import Dict, Iterable
from tqdm.notebook import tqdm
import stanza
import shutil
from pysubs2 import SSAFile, SSAEvent
from pysubs2.time import ms_to_str
from typing import List
from dataclasses import dataclass
import soundfile as sf
import pandas as pd
import numpy as np
from shutil import make_archive
from settings import Settings, default_setting
import iso639
import ffmpeg

@dataclass
class Movie:
    @dataclass
    class DownloadResult:
        @dataclass
        class DownloadBar:
            size: float
            download_link: str
            resolution: str = ''
            encoder: str = ''
            @classmethod
            def from_dict(cls, download_bar: dict):
                return cls(**download_bar)
            
        header: str
        download_bars: List[DownloadBar]
        @classmethod
        def from_dict(cls, download_result: dict):
            _download_bars = []
            if 'download_bars' in download_result:
                for _download_bar in download_result['download_bars']:
                    _download_bars.append(cls.DownloadBar.from_dict(_download_bar))
            return cls(header=download_result['header'], download_bars=_download_bars)
    
    @dataclass
    class SubtitleResult:
        subtitle_link: str
        lang: iso639.Language
        quality: str
        @classmethod
        def from_dict(cls, subtitle_result: dict):
            if subtitle_result['lang'] == "فارسی":
                _lang = iso639.Language.from_part1('fa')
            if subtitle_result['lang'] == "انگلیسی":
                _lang = iso639.Language.from_part1('en')
            else:
                _lang = None
            return cls(subtitle_link=subtitle_result['subtitle_link'], lang=_lang, quality=subtitle_result['quality'])
        
    ID: int
    title: str
    is_series: bool
    link: str
    download_results : List[DownloadResult]
    subtitle_results : List[SubtitleResult]

    @classmethod
    def from_dict(cls, movie: dict):
        _download_results = []
        _subtitle_results = []
        if 'download_results' in movie:
            for download_result in movie['download_results']:
                _download_results.append(cls.DownloadResult.from_dict(download_result))
        if 'subtitle_results' in movie:
            for subtitle_result in movie['subtitle_results']:
                _subtitle_results.append(cls.SubtitleResult.from_dict(subtitle_result))
        return cls(ID=movie['id'], title=movie['title'], is_series=movie['is_series'],
                 link=movie['link'], download_results=_download_results, subtitle_results=_subtitle_results)
    
    def __str__(self):
        return f'movie title: {self.title} ### movie id: {self.ID}'
    
    def check_movie(self):
        if len(self.download_results) == 0:
            logging.info(f"{self}\nbecause movie has 0 download_results it was dropped.")
            return False
        if len(self.download_results) < 1:
            logging.info(f"{self}\nbeacause movie has more than 1 download_results it was dropped.")
            return False
        if len(self.download_results[0].download_bars) == 0:
            logging.info(f"{self}\nbeacause movie has 0 download bars it was dropped.")
            return False
        return True
    
    def extract_title_year_from_30nama_title(self):
        if len(self.title) < 6:
            logging.warn('movie title is too short. we can not extract title and year from it.')
        _title_subdl = self.title[:-5]
        try:
            _year_subdl = int(self.title[-4:])
        except ValueError:
            raise CNamaDataException('year can not be converted to integer.')
        return _title_subdl, _year_subdl
    
    def get_min_download_bar(self, encoder : str = None):
        _minsize : int = 999_999_999
        _minsize_item = None
        _minsize_index : int = 0
        for _minsize_idx, _download_bar in enumerate(self.download_results[0].download_bars):
            if _download_bar.size < _minsize:
                if encoder:
                    if _download_bar.encoder.lower() != encoder.lower():
                        continue
                _minsize = _download_bar.size
                _minsize_item = _download_bar
                _minsize_index = _minsize_idx
        return _minsize_item, _minsize_index

    def get_best_download_bar(self):
        _minsize_item, _minsize_index = self.get_min_download_bar()
        _minsize_item_encoder_30nama, _minsize_index_encoder_30nama = self.get_min_download_bar('30nama')
        if _minsize_item_encoder_30nama:
            return _minsize_index_encoder_30nama, _minsize_item_encoder_30nama
        return _minsize_index, _minsize_item

def copy_file_with_extention(src,dst):
    original_extention = src.split('/')[-1].split('.')[-1]
    shutil.copyfile(src,dst + '.' + original_extention)
    return original_extention



@default_setting(arguments_key_idx_sname=[('sample_rate',4,'SAMPLE_RATE',)])
def convert_video_to_audio(video_file_path: Union[str, os.PathLike], audio_path: Union[str, os.PathLike], mono: bool,
                           sample_rate: int):
    output_options = {}
    if mono:
        output_options['ac'] = 1
    if sample_rate:
        output_options['ar'] = sample_rate
    ffmpeg.input(video_file_path).output(audio_path, **output_options).run()
    return audio_path

def save_json(path: Union[str, os.PathLike], obj):
    with open(path,  'w',  encoding='utf-8') as f:
        json.dump(obj,  f,  ensure_ascii=False)

def load_json(path: Union[str, os.PathLike]):
    with open(path, 'r') as f:
        _obj = json.load(f)
    return _obj

def zip_directory(directory: Union[str, os.PathLike], zip_path : Union[str, os.PathLike]):
    make_archive(zip_path.replace('.zip',''), 'zip', directory)

class ShowProgressUrllib:
    def __init__(self, description: str = 'Untitled TQDM'):
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

@default_setting(arguments_key_idx_sname=[('sample_rate',3,'SAMPLE_RATE',)])
def extract_segment_from_array(audio_array: np.array, subtitle: SSAEvent, sample_rate: int) -> np.array:
    _segment_start_index = int(subtitle.start * (sample_rate / 1000))
    _segment_end_index = int(subtitle.end * (sample_rate / 1000))
    _segment_array = audio_array[_segment_start_index: _segment_end_index]
    _min_length = int(Settings.AUDIO_MIN_LENGTH * (sample_rate))
    if _segment_array.shape[0] > _min_length:
        return _segment_array
    else:
        return np.zeros(shape=(_min_length,), dtype=np.float32)

@default_setting(arguments_key_idx_sname=[('sample_rate',3,'SAMPLE_RATE',)])
def show_parallel_data(audio_array: np.array, subtitle: SSAEvent, sample_rate: int) -> None:
    print(f'subtitle : {subtitle.content}')
    _audio_array = extract_segment_from_array(audio_array, sample_rate, subtitle)
    display(Audio(_audio_array, rate=sample_rate))


def legacy_algo(segments_list: Iterable[SSAEvent]) -> List[bool]:
    SUBTITLE_MAX_GAP = 2 # in seconds
    _is_contimous_list = []
    _index_to_be_computed = []
    _final_output_dict: Dict[int, bool] = {}
    _prompt_list = []
    for _idx, (_segment1, _segment2) in enumerate(zip(segments_list[:-1],segments_list[1:])):
        if (_segment2.start > _segment1.end + SUBTITLE_MAX_GAP):
            _final_output_dict[_idx] = False
        elif preprocess_subtitle_str(_segment1.content) and preprocess_subtitle_str(_segment2.content):
            instruction = 'Respond with either yes or no if a meaningfull bigger sentence is split between these two subtitle blocks.'
            question_prompt = 'Subtitle block : \n{}\nSubtitle block : \n{}\n\n{}'
            _a = '[' + ms_to_str(_segment1.start) + ' --> ' + ms_to_str(_segment1.end) + ']\n' + _segment1.content
            _b = '[' + ms_to_str(_segment2.start) + ' --> ' + ms_to_str(_segment2.end) + ']\n' + _segment2.content
            _prompt_str = f'{instruction}\n\n{question_prompt.format(_a,_b,"")}'
            _prompt_list.append(_prompt_str)
            _index_to_be_computed.append(_idx)
        else:
            _final_output_dict[_idx] = True
    _choices = []
    for i in range(0,len(_prompt_list),20):
        _choices.extend(Settings.OPENAI_CLIENT.completions.create(model="gpt-3.5-turbo-instruct",
                                              prompt=_prompt_list[i:i+20]
                                              , max_tokens=3, temperature=0.1).choices)
    for _idx,_choice in enumerate(_choices):
        _index_prompt = _index_to_be_computed[_idx]
        _response = _choice.text.replace('\n', '')
        _response = Settings.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _response)
        if _response.lower() == 'yes':
            _final_output_dict[_index_prompt] = True
        else:
            _final_output_dict[_index_prompt] = False
    for i in range(len(segments_list)-1):
        _is_contimous_list.append(_final_output_dict[i])
    return _is_contimous_list

def preprocess_subtitle_str(subtitle_str: str) -> str:
    _subtitle_str = subtitle_str.replace('\u200c',' ')
    _subtitle_str = _subtitle_str.replace('\n',' ')
    _subtitle_str = Settings.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _subtitle_str)
    _subtitle_str = Settings.PARANTHESIS_REGEX.sub('', _subtitle_str)
    _subtitle_str = Settings.CONTINOUS_SYMBOL_REGEX.sub('', _subtitle_str)
    _subtitle_str = Settings.LEADING_TRAILING_WHITESPACE_REGEX.sub('', _subtitle_str)
    return _subtitle_str

def zip_wavs(movie: Movie):
    with zipfile.ZipFile(f'./dataset/{movie.ID}/{movie.ID}.zip', 'w', zipfile.ZIP_DEFLATED) as _zipf:
        _wavs = glob.glob(f'./dataset/{movie.ID}/*.mp3')
        for _wav in _wavs:
            _zipf.write(_wav, _wav.split('/')[-1], compress_type=zipfile.ZIP_DEFLATED)
            os.remove(_wav)

@default_setting(arguments_key_idx_sname=[('sample_rate', 8, 'SAMPLE_RATE',)])
def save_results(movie: Movie, audio_array: np.array, movie_in_minutes,
                 movie_in_seconds, final_subtitles: Iterable[SSAEvent], subtitle_lang: str,
                 similarities: np.array, sample_rate: float):
    pd_dict = {}
    pd_dict['start'] = [subtitle.start / 1000 for subtitle in final_subtitles]
    pd_dict['end'] = [subtitle.end / 1000 for subtitle in final_subtitles]
    pd_dict['content'] = [subtitle.plaintext for subtitle in final_subtitles]
    pd_dict['sub_index'] = [i for i in range(len(final_subtitles))]
    pd_dict['lang'] = [subtitle_lang]*len(final_subtitles)
    pd_dict['movie_id'] = [movie.ID]*len(final_subtitles)
    pd_dict['movie_duration_in_minutes'] = [movie_in_minutes]*len(final_subtitles)
    pd_dict['movie_duration_in_seconds'] = [movie_in_seconds]*len(final_subtitles)
    pd_dict['segment_duration'] = [(subtitle.end - subtitle.start) / 1000 for subtitle in final_subtitles]
    pd_dict['score'] = similarities.detach().cpu().numpy()
    df = pd.DataFrame.from_dict(pd_dict, 'columns')
    df = df.sort_values('score', ascending=False).drop_duplicates(['movie_id','sub_index'],keep='first')
    segment_path_list = []
    for index, row in df.iterrows():
        segment_array = audio_array[int(row['start']*sample_rate): int(row['end']*sample_rate)]
        sf.write(f'./dataset/{movie.ID}/{row["sub_index"]}.mp3', segment_array, sample_rate, format='wav')
        segment_path_list.append(f'{movie.ID}/{row["sub_index"]}.mp3')
    df['path'] = segment_path_list
    df.to_csv(f'./dataset/{subtitle_lang}_{movie.ID}.csv', index=False)
    zip_wavs(movie.ID)


def predict_encoding(file_path: Path, n_lines: int=20) -> str:
    '''Predict a file's encoding using chardet'''

    if file_path.find('[ANSI]') != -1:
        return 'cp1256'

    # Open the file as binary data
    with Path(file_path).open('rb') as f:
        # Join binary lines for specified number of lines
        _rawdata = b''.join([f.readline() for _ in range(n_lines)])

    return chardet.detect(_rawdata)['encoding']

def divide_subtitle_list(subtitle_list: Iterable[SSAEvent], continous_list: Iterable[bool]) -> Iterable[Iterable[SSAEvent]]:
    _subtitle_division_list = []
    _division = []
    if len(continous_list) != len(subtitle_list) - 1:
        raise SubtitleProcessingException(f'Subtitle continous list length should be {len(subtitle_list) - 1} but it is {len(continous_list)}')
    for _subtitle, _is_continous in zip(subtitle_list, [True] + continous_list):
        if _division and (not _is_continous):
            _subtitle_division_list.append(_division)
            _division = []
        _division.append(_subtitle)    
    if _division:
        _subtitle_division_list.append(_division)
    return _subtitle_division_list

@default_setting(arguments_key_idx_sname=[('lid_model',2,'LID_PIPELINE',)])
def detect_lang(subtitles: SSAFile, lid_model: stanza.pipeline.core.Pipeline=None):
    concat_all_content = ''
    for sub in subtitles:
        concat_all_content += ' ' + sub.plaintext 
    return iso639.Language.from_part1(lid_model(concat_all_content).lang)

def process_subtitle_file(subtitle_list: Iterable[SSAEvent], continous_fn=legacy_algo, preprocess_fn=preprocess_subtitle_str) -> Iterable[SSAEvent]:
    _processed_subtitles = []
    _continous_truth_list = continous_fn(subtitle_list)
    _subtitle_division_list = divide_subtitle_list(subtitle_list,_continous_truth_list)
    for  _division in _subtitle_division_list:
        _start = _division[0].start
        _end = _division[-1].end
        _division_content = ''
        for _subtitle in _division:
            _division_content += preprocess_fn(_subtitle.plaintext) + ' '
        _division_content = preprocess_fn(_division_content)
        _new_subtitle = SSAEvent(start=_start, end=_end)
        _new_subtitle.plaintext = _division_content
        _processed_subtitles.append(_new_subtitle)
    return _processed_subtitles

class SubtitleProcessingException(BaseException):
    def __init__(self, message) -> None:
        super().__init__(message)

class CNamaDataException(Exception):
    def __init__(self, message) -> None:
        super(Exception, self).__init__(message)
