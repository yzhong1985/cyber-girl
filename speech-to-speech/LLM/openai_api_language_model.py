import logging
import time
import re

from nltk import sent_tokenize
from rich.console import Console
from openai import OpenAI

from baseHandler import BaseHandler
from LLM.chat import Chat

logger = logging.getLogger(__name__)

console = Console()

# Tool call pattern for extraction
TOOL_PATTERN = re.compile(r'\[TOOL:(\w+)(?:\|([^\]]+))?\]')

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}


def parse_tool_calls(text):
    """Extract tool calls from text and return cleaned text + tool calls."""
    tools = []

    for match in TOOL_PATTERN.finditer(text):
        tool_name = match.group(1)
        params_str = match.group(2) or ""

        # Parse params: key1:val1|key2:val2
        params = {}
        if params_str:
            for param in params_str.split('|'):
                if ':' in param:
                    key, val = param.split(':', 1)
                    params[key] = val

        tools.append({"name": tool_name, "parameters": params})

    # Remove tool markers from text
    clean_text = TOOL_PATTERN.sub('', text).strip()

    return clean_text, tools


class OpenApiModelHandler(BaseHandler):
    """
    Handles the language model part.
    """
    def setup(
        self,
        model_name="deepseek-chat",
        device="cuda",
        gen_kwargs={},
        base_url =None,
        api_key=None,
        stream=False,
        user_role="user",
        chat_size=1,
        init_chat_role="system",
        init_chat_prompt="You are a helpful AI assistant.",
        registry=None,
    ):
        self.model_name = model_name
        self.stream = stream
        self.chat = Chat(chat_size)
        self.init_chat_role = init_chat_role
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # 多角色管理: 切角色时把 system_prompt 换成新角色的, 并清空对话历史
        # (不然新角色会"记得"上一个角色聊过的东西, 很奇怪)。之前这里完全没接,
        # 只有音色(TTS)会跟着热切换, 人格是死的。
        self.registry = registry
        if self.registry is not None:
            self.registry.register_callback(self._on_persona_switch)
        self.warmup()

    def _on_persona_switch(self, name):
        try:
            new_prompt = self.registry.current_prompt()
        except Exception as e:
            logger.warning(f"[LLM] 切角色时读取新 system_prompt 失败(不致命): {e}")
            return
        self.chat.init_chat({"role": self.init_chat_role, "content": new_prompt})
        self.chat.buffer = []
        logger.info(f"[LLM] 已切到角色「{name}」的人格, 对话历史已清空")

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        # 用真实角色的 system_prompt(而不是通用占位提示词)预热, 这样服务端(llama.cpp)
        # 的 prompt 前缀缓存命中的就是正式对话真正会用到的前缀, 消除首轮对话额外的
        # 冷启动 prefill 延迟(实测: 冷=~4~8s TTFT, 命中缓存=~0.1~0.3s)。不写回
        # self.chat, 不污染正式对话历史。
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.chat.to_list() + [{"role": self.user_role, "content": "你好"}],
            stream=self.stream
        )
        end = time.time()
        logger.info(
            f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s"
        )
    def process(self, prompt):
            logger.debug("call api language model...")

            language_code = None
            if isinstance(prompt, tuple):
                prompt, language_code = prompt
                if language_code[-5:] == "-auto":
                    language_code = language_code[:-5]
                    prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

            self.chat.append({"role": self.user_role, "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.chat.to_list(),
                stream=self.stream
            )
            if self.stream:
                generated_text, printable_text = "", ""
                for chunk in response:
                    new_text = chunk.choices[0].delta.content or ""
                    generated_text += new_text
                    printable_text += new_text
                    sentences = sent_tokenize(printable_text)
                    if len(sentences) > 1:
                        clean_text, tools = parse_tool_calls(sentences[0])
                        yield clean_text, language_code, tools
                        printable_text = new_text
                self.chat.append({"role": "assistant", "content": generated_text})
                # don't forget last sentence
                clean_text, tools = parse_tool_calls(printable_text)
                yield clean_text, language_code, tools
            else:
                generated_text = response.choices[0].message.content
                self.chat.append({"role": "assistant", "content": generated_text})
                clean_text, tools = parse_tool_calls(generated_text)
                yield clean_text, language_code, tools

