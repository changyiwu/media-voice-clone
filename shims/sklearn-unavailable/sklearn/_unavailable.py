"""讓「import 成功、用到才報錯」的模組層 __getattr__。"""

MESSAGE = (
    'scikit-learn 在這台無法使用：Smart App Control 封鎖了它的 win_arm64 原生模組，'
    '目前裝的是 media-voice-clone 的 sklearn-unavailable 空殼（shims/sklearn-unavailable）。'
    '語音生成用不到它；需要 scikit-learn 的功能請改用 x64 環境。'
)


def make_getattr(module_name):
    def __getattr__(name):
        # 匯入機制與各種工具會試探 __path__、__wrapped__ 之類的屬性，照常回 AttributeError
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f'module {module_name!r} has no attribute {name!r}')
        raise ImportError(f'{module_name}.{name}：{MESSAGE}')
    return __getattr__
