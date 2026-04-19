from utils.config_handler import prompts_conf
from utils.path_tools import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf['main_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml中未找到配置项main_prompt_path")
        raise e

    try:
        return open(system_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_system_prompts]加载系统提示词文件失败，错误信息：{str(e)}")
        raise e

def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf['rag_summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml中未找到配置项rag_summarize_prompt_path")
        raise e

    try:
        return open(rag_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]加载RAG提示词文件失败，错误信息：{str(e)}")
        raise e

def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf['report_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml中未找到配置项report_prompt_path")
        raise e

    try:
        return open(report_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_report_prompts]加载Report提示词文件失败，错误信息：{str(e)}")
        raise e