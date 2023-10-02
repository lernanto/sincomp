# -*- coding: utf-8 -*-

"""
音系分析相关功能函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import ipapy.ipachar
import ipapy.ipastring


def get_voicing(s):
    """
    返回声母的清浊.

    Parameters:
        s (str): 声母 IPA 字符串

    Returns:
        type (str): 声母的类型，可能的取值为：
            - voiced: 浊音
            - voiceless: 清音
            - None: s 首个字母不是辅音字母

    取 s 第一个辅音字母及其辅标的清浊为整个声母的清浊。
    """

    # 先把中国方言学界特有的国际音标符号转换成近似的标准国际音标，否则不能识别
    s = s.translate({
        0x0235: 0x0272,     # ȵ
        0x0236: 0x0063,     # ȶ
    })

    ipa = ipapy.ipastring.IPAString(unicode_string=s, ignore=True)

    # 截取首个字母及其辅标
    i = 0
    while i < len(ipa) and not ipa[i].is_letter:
        i += 1
    i += 1
    while i < len(ipa) and not ipa[i].is_letter:
        i += 1

    ipa = ipapy.ipastring.IPAString(ipa[:i])
    if len(ipa.consonants) == 0:
        # 首个字母不是辅音字母
        return None

    return 'voiced' if 'voiced' in [c.dg_value(ipapy.ipachar.DG_C_VOICING) for c in ipa] \
        else 'voiceless'

def get_manner(s):
    """
    返回声母的发音方法.

    Paramters:
        s (str): 声母 IPA 字符串

    Returns:
        manner (str): 声母的发音方法

    取最后一个辅音字母的发音方法。
    """

    s = s.translate({
        0x0235: 0x0272,     # ȵ
        0x0236: 0x0063,     # ȶ
    })

    ipa = ipapy.ipastring.IPAString(unicode_string=s, ignore=True)
    if len(ipa.consonants) == 0:
        return None

    return ipa.consonants[-1].dg_value(ipapy.ipachar.DG_C_MANNER).split('-')[-1]

def get_coda_type(s):
    """
    返回韵尾的类型.

    Parameters:
        s (str): 韵母 IPA 字符串

    Returns:
        type (str): 韵尾的类型，可能的取值为：
            - vowel: 元音
            - nasal: 鼻音
            - nasalized: 鼻化元音或辅音
            - 其他 get_manner 返回的发音方法名称
            - None: 输入的字符串不包含 IPA 字母

    取最后一个 IPA 字母的发音方法。
    """

    s = s.translate({
        0x0235: 0x0272,     # ȵ
        0x0236: 0x0063,     # ȶ
        0x0277: 0x028a,     # ɷ
        0x027f: 0x0279,     # ɿ
        0x0285: 0x027b,     # ʅ
        0x02ae: 0x0279,     # ʮ
        0x02af: 0x027b,     # ʯ
        0x1d00: 0x0061,     # ᴀ
        0x1d07: 0x0065,     # ᴇ
    })

    ipa = ipapy.ipastring.IPAString(unicode_string=s, ignore=True)
    if len(ipa.letters) == 0:
        return None

    return 'nasalized' if ipa[-1].dg_value(ipapy.ipachar.DG_DIACRITICS) == 'nasalized' \
        else 'vowel' if ipa.letters[-1].is_vowel else get_manner(s)