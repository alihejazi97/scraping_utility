import torch
from torch.nn import CosineSimilarity
import re
from typing import List, Dict, Union
import urllib.request
from iso639 import Language
from stanza.pipeline.core import Pipeline

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWeb'),
            ('User-agent', 'Kit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safa'),
            ('User-agent', 'ri/537.36')]
urllib.request.install_opener(opener)

def default_setting(arguments_key_idx_sname):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for argument_key, argument_index, setting_name in arguments_key_idx_sname:
                if argument_index > len(args):
                    if argument_key not in kwargs:
                        kwargs[argument_key] = Settings.__getattribute__(Settings, setting_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Settings:  
    SAMPLE_RATE: float = 16_000
    SUBDL_SLEEP_TIME: float = 1
    SIMILARITY_MEASURE = CosineSimilarity(dim=1, eps=1e-6)
    AUDIO_MIN_LENGTH: float = 0.2 # in seconds
    OPENAI_API_KEYS: Union[str, List[str]] = None
    SUBDL_API_KEYS: Union[str, List[str]] = None
    SPEECH_EMBEDDING_COMPUTATION_BATCH_SIZE: int = 4
    TEXT_EMBEDDING_COMPUTATION_BATCH_SIZE: int = 8
    LID_PIPELINE: Pipeline = None
    DEBUG_VAR = None
    PARANTHESIS_REGEX_STR_LIST: List[str] = [r'(\[[\S\s]*\])',r'(\][\S\s]*\[)',r'(\)[\S\s]*\()',r'(\([\S\s]*\))']
    PARANTHESIS_REGEX: re.Pattern = re.compile(r'|'.join(PARANTHESIS_REGEX_STR_LIST))
    LEADING_TRAILING_WHITESPACE_REGEX: re.Pattern = re.compile(r'^\s+|\s+$')
    CONTINOUS_SYMBOL_REGEX: re.Pattern = re.compile(r'(^(\.\.\.))|(((\.\.\.)|:)$)')
    SUBDL_SEARCH_MOVIE_URL: str = 'https://api.subdl.com/auto'
    SUBDL_SEARCH_SUBTITLE_URL: str = 'https://api.subdl.com/api/v1/subtitles'
    SUBDL_SUBTITLE_DOWNLOAD_URL: str = 'https://dl.subdl.com'
    
    PERSIAN = Language.from_part1('fa')
    ENGLISH = Language.from_part1('en')
    GERMAN = Language.from_part1('de')
    FRENCH = Language.from_part1('fr')
    SPANISH = Language.from_part1('es')
    ITALIAN = Language.from_part1('it')
    ARABIC = Language.from_part1('ar')
    PORTUGUESE = Language.from_part1('pt')
    RUSSIAN = Language.from_part1('ru')
    TURKISH = Language.from_part1('tr')
    UKRAINIAN = Language.from_part1('uk')
    POLISH = Language.from_part1('pl')
    CHINESE = Language.from_part1('zh')
    HINDI = Language.from_part1('hi')
    BENGALI = Language.from_part1('bn')
    JAPANESE = Language.from_part1('ja')
    VIETNAMESE = Language.from_part1('vi')
    KOREAN = Language.from_part1('ko')
    FINNISH = Language.from_part1('fi')
    URDU = Language.from_part1('ur')
    UNKNOWN_LANGUAGE = Language.from_part1('vo')
    SONAR_TEXT_LANG_MAP: Dict[Language, str] = {
        PERSIAN : 'sonar_speech_encoder_base_pes', # persian
        ENGLISH : 'sonar_speech_encoder_base_eng', # english
        GERMAN : 'sonar_speech_encoder_base_deu', # german
        FRENCH : 'sonar_speech_encoder_base_fra', # france
        SPANISH : 'sonar_speech_encoder_base_spa', # spanish
        ITALIAN : 'sonar_speech_encoder_base_ita', # italian
        ARABIC : 'sonar_speech_encoder_base_arb', # arabic
        PORTUGUESE : 'sonar_speech_encoder_base_por', # portuguese
        RUSSIAN : 'sonar_speech_encoder_base_rus', # russian
        TURKISH : 'sonar_speech_encoder_base_tur', # turkish
        UKRAINIAN : 'sonar_speech_encoder_base_ukr', # ukrainian
        POLISH : 'sonar_speech_encoder_base_pol', # polish
        CHINESE : 'sonar_speech_encoder_base_cmn', # chinese
        HINDI : 'sonar_speech_encoder_base_hin', # hindi
        BENGALI : 'sonar_speech_encoder_base_ben', # bengali
        JAPANESE : 'sonar_speech_encoder_base_jpn', # japanese
        VIETNAMESE : 'sonar_speech_encoder_base_vie', # vietnamese
        KOREAN : 'sonar_speech_encoder_base_kor', # korean
        FINNISH : 'sonar_speech_encoder_base_fin', # finnish
        URDU : 'sonar_speech_encoder_base_urd'  # Urdu
    }

    SONAR_SPEECH_ENCODER_LANG_MAP : Dict[Language, str] = {
        PERSIAN : 'pes_Arab',      # persian
        ENGLISH : 'eng_Latn',      # english
        GERMAN : 'deu_Latn',      # german
        FRENCH : 'fra_Latn',      # france
        SPANISH : 'spa_Latn',      # spanish
        ITALIAN : 'ita_Latn',      # italian
        ARABIC : 'arb_Arab',      # arabic
        PORTUGUESE : 'por_Latn',      # portuguese
        RUSSIAN : 'rus_Cyrl',      # russian
        TURKISH : 'tur_Latn',      # turkish
        UKRAINIAN : 'ukr_Cyrl',      # ukrainian
        POLISH : 'pol_Latn',      # polish
        CHINESE : 'zho_Hans',      # chinese
        HINDI : 'hin_Deva',      # hindi
        BENGALI : 'ben_Beng',      # bengali
        JAPANESE : 'jpn_Jpan',      # japanese
        VIETNAMESE : 'vie_Latn',      # vietnamese
        KOREAN : 'kor_Hang',      # korean
        FINNISH : 'fin_Latn',      # finnish
        URDU : 'urd_Arab'       # Urdu
    }

    SUBDL_LANG_CODES : Dict[Language, List[str]] = {
        PERSIAN : ['FA'],          # persian
        ENGLISH : ['EN'],          # english
        GERMAN : ['DE', 'EN_DE'], # german
        FRENCH : ['FR'],          # france
        SPANISH : ['ES'],          # spanish
        ITALIAN : ['IT'],          # italian
        ARABIC : ['AR'],          # arabic
        PORTUGUESE : ['PT'],          # portuguese
        RUSSIAN : ['RU'],          # russian
        TURKISH : ['TR'],          # turkish
        UKRAINIAN : ['UK'],          # ukrainian
        POLISH : ['PL'],          # polish
        CHINESE : ['ZH_BG', 'ZH'], # chinese
        HINDI : ['HI'],          # hindi
        BENGALI : ['BN'],          # bengali
        JAPANESE : ['JA'],          # japanese
        VIETNAMESE : ['VI'],          # vietnamese
        KOREAN : ['KO'],          # korean
        FINNISH : ['FI'],          # finnish
        URDU : ['UR']           # Urdu
    }

    SUBDL_LANGUAGE_MAP : Dict[str, Language] = {
    "farsi_persian" : PERSIAN,     # persian
    "english" : ENGLISH,           # english
    "english_german" : GERMAN,    # german
    "german" : GERMAN,            # german
    "french" : FRENCH,            # france
    "spanish" : SPANISH,           # spanish
    "italian" : ITALIAN,           # italian
    "arabic" : ARABIC,            # arabic
    "portuguese" : PORTUGUESE,        # portuguese
    "russian" : RUSSIAN,           # russian
    "turkish" : TURKISH,           # turkish
    "ukranian" : UKRAINIAN,          # ukrainian
    "polish": POLISH,             # polish
    "big 5 code": CHINESE,         # chinese
    "chinese bg code" : CHINESE,   # chinese
    "hindi" : HINDI,             # hindi
    "bengali" : BENGALI,           # bengali
    "japanese" : JAPANESE,          # japanese
    "vietnamese" : VIETNAMESE,        # vietnamese
    "korean" : KOREAN,            # korean
    "finnish" : FINNISH,           # finnish
    "urdu" : URDU               # Urdu
    }