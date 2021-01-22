# coding=utf-8
import sys

__author__ = 'cha'

import re
import os
import subprocess
from collections import OrderedDict

MAX_MORPH_LENGTH = 45
MAX_SYLL_LENGTH = 45
MAX_CHAR_LENGTH = 45
NUM_MORPH_PAD = 2
NUM_SYLL_PAD = 2
NUM_CHAR_PAD = 2

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(br"\d")


def get_morphs_tags(morphs_str, sep=' + '):
    morphs_tags = []

    for morph_str in morphs_str.split(sep):
        match = re.match(re.compile(r'(.+?)(__[0-9]{2})?/(.+)'), morph_str)

        if match is not None:
            morphs_tags.append((match.group(1).strip(), match.group(3).strip()))
        else: # malformed data - no morph tag
            morphs_tags.append((morph_str, 'UNK'))

    return morphs_tags


def get_morphs(morphs_tags):
    result_morphs = []

    for morph, tag in morphs_tags:
        result_morphs.append('/'.join([morph, tag]))

    return result_morphs


def get_sylls(morphs_tags):
    result_sylls = []

    for morph, tag in morphs_tags:
        if morph == '<ROOT>':
            result_sylls.append('/'.join([morph, tag]))

        else:
            for i, char in enumerate(morph):
                if i == 0:
                    char_tag = 'B-' + tag
                else:
                    char_tag = 'I-' + tag

                result_sylls.append('/'.join([char, char_tag]))

    return result_sylls


def split_plus(plus_joined_str, sep='+'):
    """
    return words split by '+'
    this method can process strings including '+' words
    ex) '+++' -> ['+', '+']
    """

    plus_joined_str += sep
    return [match.group(1) for match in re.finditer(r'(.+?)' + re.escape(sep), plus_joined_str)]


def split_raw(raw):
    # QN
    if re.match(r'Q[0-9]', raw) is not None:
        raw = raw[2:]

    # pairs: '' 『』 「」 "" () [] {} <>
    pair_exp_format = '({0}[^{0}{1}]+?{1})'
    pair_exp = []

    pairs = [('\'', '\''), ('『', '』'), ('「', '」'), ('"', '"'), ('\\(', '\\)'), ('\\[', '\\]'), ('{', '}'), ('<', '>')]

    for pair in pairs:
        pair_exp = pair_exp_format.format(pair[0], pair[1])
        matches = re.finditer(pair_exp, raw)
        changes = []

        for match in matches:
            if re.findall(r' ', raw[match.start():match.end()]):
                original = raw[match.start():match.end()]
                changed = ' '.join(['', original[0], original[1:-1], original[-1], ''])
                changes.append((original, changed))

        for change in changes:
            raw = raw.replace(change[0], change[1])

    # special: - ─ ] /
    raw = re.sub(r'([\-─/])', r' \1 ', raw)

    # force no space before: .
    raw = re.sub(r' ([.])', r'\1', raw)

    return raw.split()


def get_mor_result_v2(sentence):
    # korea univ morpheme analyzer
    # dict 에 두번 들어가는 경우.
    # 문장에 + 들어가는 경우도 처리해줘야 함.
    # 문장에 '' < 있는 경우도 처리해 줘야함.

    try:
        m_command = "cd ../data/kmat/bin/;./kmat <<<\'" + sentence + "\' 2>/dev/null"
        with open(os.devnull, 'w') as devnull:
            result = subprocess.check_output(m_command.encode(encoding='cp949', errors='ignore'), shell=True,
                                            executable='/bin/bash', stderr=devnull)
    except:
        return None

    mor_name_lists = []
    mor_tags_lists = []
    mor_dict = OrderedDict()

    count = 0
    for each in result.decode(encoding='cp949', errors='ignore').split('\n'):
        if len(each) > 0:
            try:
                ori_text = each.split('\t')[0]
                mor_texts = each.split('\t')[1]
                mor_results = mor_texts.split('+')

                count += 1

                dict_key = ori_text
                if dict_key in mor_dict:
                    dict_key = dict_key + '||' + str(count)
                mor_dict[dict_key] = []
                for each_mor in mor_results:
                    try:
                        if not each_mor.strip():
                            del mor_dict[dict_key]
                            break

                        mor_name = each_mor.split('/')[0]
                        mor_tags = each_mor.split('/')[1]

                        if not mor_name or not mor_tags:
                            del mor_dict[dict_key]
                            break

                        mor_name_lists.append(mor_name)
                        mor_tags_lists.append(mor_tags)
                        each_mor_dict = {}
                        each_mor_dict[mor_name] = mor_tags

                        mor_dict[dict_key].append(each_mor_dict)
                    except Exception as e:
                        print(e)
                        print(each_mor)
            except:
                print(each)

    return mor_dict