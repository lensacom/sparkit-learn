# -*- coding: utf-8 -*-


def flesch_kincaid_grade(sentence_count, word_count, syllable_count):
    if word_count == 0 or sentence_count == 0:
        return .0
    else:
        return .39 * (float(word_count) / sentence_count) + \
            11.8 * (float(syllable_count) / word_count) - 15.59
