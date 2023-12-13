import translators as ts
from translators.server import TranslatorError

def translate(q_text, ts_tool: str = 'google', ts_lang: str = 'ko', from_lang='en'):
    result = ts.translate_text(q_text, translator=ts_tool, if_use_preacceleration=False, is_detail_result=False, timeout=10, to_language=ts_lang, from_language=from_lang)
    return result

translators_pool = ts.translators_pool

def get_languages(ts_tool):
    return ts.get_languages(ts_tool)