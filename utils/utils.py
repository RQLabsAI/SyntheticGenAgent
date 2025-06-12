import math
from collections import Counter
from typing import List

from models.models import Document, DatasetMetrics

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_n_grams(items: list[str], n: int) -> list[str]:
    return [" ".join(items[i: i + n]) for i in range(len(items) - n + 1)]


def format_report(metrics: DatasetMetrics, with_domains=True):
    output = f"""# Dataset metrics
## Total
Number of documents present in the dataset: {metrics.number_of_docs}
Number of words in the dataset: {metrics.dataset_total_words}
Average words per document: {metrics.dataset_average_words_per_document}
Average sentence length: {round(metrics.dataset_average_sentence_length, 4)} words
Median of sentence length: {round(metrics.dataset_median_sentence_length, 4)} words
Standard deviation: {round(metrics.dataset_std_sentence_length, 4)} words
Shannon entropy: {round(metrics.dataset_shannon_entropy, 4)}
## Number of documents in each domain
{metrics.domains_counter}
## Dataset most common Part of Speech
{metrics.dataset_most_common_pos}
## Dataset most common words
{metrics.dataset_most_common_words}
## Dataset least common words
{metrics.dataset_least_common_words}
## Dataset special characters to alphanumeric characters ratio
{metrics.dataset_special_characters_to_alpha_ratio}
## Most frequent 2-grams
{metrics.dataset_most_frequent_2_grams}
## Most frequent 3-grams
{metrics.dataset_most_frequent_3_grams}
## Most frequent 4-grams
{metrics.dataset_most_frequent_4_grams}
## Most frequent 5-grams
{metrics.dataset_most_frequent_5_grams}
## Most frequent 6-grams
{metrics.dataset_most_frequent_6_grams}
## Dataset top TF–IDF terms
{metrics.dataset_top_tfidf}
"""

    if with_domains:
        output += "\n# Per domain metric\n"
        for domain in metrics.domain_most_common_words.keys():
            output += f"""## Domain: {domain}
### Total metrics for domain {domain}
Number of words in the dataset: {metrics.domain_total_words[domain]}
Average words per document: {metrics.domain_average_words_per_document[domain]}
Average sentence length: {round(metrics.domain_average_sentence_length[domain], 4)} words
Median of sentence length: {round(metrics.domain_median_sentence_length[domain], 4)} words
Standard deviation: {round(metrics.domain_std_sentence_length[domain], 4)} words
Shannon entropy: {round(metrics.domain_shannon_entropy[domain], 4)}
### Most common Part of Speech
{metrics.domain_most_common_pos[domain]}
### Most common words
{metrics.domain_most_common_words[domain]}
### Least common words
{metrics.domain_least_common_words[domain]}
### Special characters to alphanumeric characters ratio
{metrics.domain_special_characters_to_alpha_ratio[domain]}
### Most frequent 2-grams
{metrics.domain_most_frequent_2_grams[domain]}
### Most frequent 3-grams
{metrics.domain_most_frequent_3_grams[domain]}
### Most frequent 4-grams
{metrics.domain_most_frequent_4_grams[domain]}
### Most frequent 5-grams
{metrics.domain_most_frequent_5_grams[domain]}
### Most frequent 6-grams
{metrics.domain_most_frequent_6_grams[domain]}
"""
    return output




def get_dataset_metrics(docs: List[Document]) -> DatasetMetrics:
    # Prepare text corpus and domain labels for silhouette
    corpus = []
    labels = []
    domain_to_idx = {}
    for doc in docs:
        tokens = [t.token for t in doc.tokens if t.pos != "PUNCT" and t.token not in ("\n","\n\n")]
        corpus.append(" ".join(tokens))
        if doc.domain not in domain_to_idx:
            domain_to_idx[doc.domain] = len(domain_to_idx)
        labels.append(domain_to_idx[doc.domain])

    # TF–IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # Silhouette score (cosine)
    # sil_score = silhouette_score(tfidf_matrix, labels, metric='cosine')

    # Compute average TF–IDF
    avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    top_tfidf_idxs = np.argsort(avg_tfidf)[::-1][:20]
    dataset_top_tfidf = {feature_names[i]: float(round(avg_tfidf[i],4)) for i in top_tfidf_idxs}

    # Initialize counters
    pos_counter = Counter()
    word_counter = Counter()
    total_words = 0
    sentence_lengths = []
    unique_words_to_all_words = []
    domains_counter = Counter()

    pos_counter_per_domain = {}
    word_counter_per_domain = {}
    total_words_per_domain = {}
    sentence_lengths_per_domain = {}
    unique_words_to_all_words_domain = {}

    words_per_doc = []
    words_per_doc_per_domain = {}

    entropy = 0
    entropy_per_domain = {}

    non_alphanumeric_characters = []
    non_alphanumeric_characters_per_domain = {}

    all_2_grams_counter = Counter()
    all_3_grams_counter = Counter()
    all_4_grams_counter = Counter()
    all_5_grams_counter = Counter()
    all_6_grams_counter = Counter()

    domain_2_grams_counters = {}
    domain_3_grams_counters = {}
    domain_4_grams_counters = {}
    domain_5_grams_counters = {}
    domain_6_grams_counters = {}

    dataset_target_audience_counter = Counter()
    dataset_style_counter = Counter()

    domain_target_audience_counter = {}
    domain_style_counter = {}

    for doc in docs:
        # global lists
        poss = [token.pos for token in doc.tokens]
        words = [token.token for token in doc.tokens]
        words_without_puncts = [token.token for token in doc.tokens if
                                token.pos != "PUNCT" and token.token != "\n" and token.token != "\n\n"]
        sentences_lens = [len(sent) for sent in doc.sentences]
        unique_words_ratio = len(set(words)) / len(words)

        domains_counter[doc.domain] += 1

        # update metrics for the whole dataset
        pos_counter.update(poss)
        word_counter.update(words_without_puncts)
        total_words += len(doc.tokens)
        unique_words_to_all_words.append(unique_words_ratio)

        words_per_doc.append(len(doc.tokens))

        sentence_lengths.extend(sentences_lens)

        non_alpha_chars_ratio = sum([any((c.isalpha() for c in str(w))) for w in doc.tokens]) / len(words)

        non_alphanumeric_characters.append(non_alpha_chars_ratio)

        # create objects for each domain
        if doc.domain not in domain_target_audience_counter:
            domain_target_audience_counter[doc.domain] = Counter()
        domain_target_audience_counter[doc.domain][doc.target_audience] += 1

        if doc.domain not in domain_style_counter:
            domain_style_counter[doc.domain] = Counter()
        domain_style_counter[doc.domain][doc.style] += 1

        if doc.domain not in pos_counter_per_domain:
            pos_counter_per_domain[doc.domain] = Counter()

        if doc.domain not in word_counter_per_domain:
            word_counter_per_domain[doc.domain] = Counter()

        if doc.domain not in total_words_per_domain:
            total_words_per_domain[doc.domain] = 0

        if doc.domain not in words_per_doc_per_domain:
            words_per_doc_per_domain[doc.domain] = []

        if doc.domain not in sentence_lengths_per_domain:
            sentence_lengths_per_domain[doc.domain] = []

        if doc.domain not in unique_words_to_all_words_domain:
            unique_words_to_all_words_domain[doc.domain] = []

        if doc.domain not in non_alphanumeric_characters_per_domain:
            non_alphanumeric_characters_per_domain[doc.domain] = []

        if doc.domain not in domain_2_grams_counters:
            domain_2_grams_counters[doc.domain] = Counter()

        if doc.domain not in domain_3_grams_counters:
            domain_3_grams_counters[doc.domain] = Counter()

        if doc.domain not in domain_4_grams_counters:
            domain_4_grams_counters[doc.domain] = Counter()

        if doc.domain not in domain_5_grams_counters:
            domain_5_grams_counters[doc.domain] = Counter()

        if doc.domain not in domain_6_grams_counters:
            domain_6_grams_counters[doc.domain] = Counter()

        str_tokens = words_without_puncts
        for n2gram in get_n_grams(str_tokens, 2):
            all_2_grams_counter[n2gram] += 1
            domain_2_grams_counters[doc.domain][n2gram] += 1

        for n3gram in get_n_grams(str_tokens, 3):
            all_3_grams_counter[n3gram] += 1
            domain_3_grams_counters[doc.domain][n3gram] += 1

        for n4gram in get_n_grams(str_tokens, 4):
            all_4_grams_counter[n4gram] += 1
            domain_4_grams_counters[doc.domain][n4gram] += 1

        for n5gram in get_n_grams(str_tokens, 5):
            all_5_grams_counter[n5gram] += 1
            domain_5_grams_counters[doc.domain][n5gram] += 1

        for n6gram in get_n_grams(str_tokens, 6):
            all_6_grams_counter[n6gram] += 1
            domain_6_grams_counters[doc.domain][n6gram] += 1

        # update metrics for each domain
        pos_counter_per_domain[doc.domain].update(poss)
        word_counter_per_domain[doc.domain].update(words)
        total_words_per_domain[doc.domain] += len(words)
        words_per_doc_per_domain[doc.domain].append(len(doc.tokens))
        sentence_lengths_per_domain[doc.domain].extend(sentences_lens)
        unique_words_to_all_words_domain[doc.domain].append(unique_words_ratio)
        non_alphanumeric_characters_per_domain[doc.domain].append(non_alpha_chars_ratio)

    # entropy for the whole dataset
    for word, count in word_counter.items():
        probability = count / total_words
        entropy -= probability * math.log2(probability)

    for domain in total_words_per_domain.keys():
        if domain not in entropy_per_domain:
            entropy_per_domain[domain] = 0

        for word, count in word_counter_per_domain[domain].items():
            probability = count / total_words
            entropy_per_domain[domain] -= probability * math.log2(probability)

        entropy_per_domain[domain] = round(entropy_per_domain[domain], 4)

    entropy = round(entropy, 4)
    avg_words_per_doc = float(np.mean(words_per_doc))
    avg_sentence_lengths = float(np.mean(sentence_lengths))
    median_sentence_lengths = int(np.median(sentence_lengths))
    std_sentence_lengths = float(np.std(sentence_lengths))

    domain_most_common_pos = {}
    for domain, cnt in pos_counter_per_domain.items():
        domain_most_common_pos[domain] = dict(cnt.most_common(10))

    domain_most_common_words = {}
    for domain, cnt in word_counter_per_domain.items():
        domain_most_common_words[domain] = dict(cnt.most_common(20))

    domain_least_common_words = {}
    for domain, cnt in word_counter_per_domain.items():
        domain_least_common_words[domain] = dict(cnt.most_common()[::-20])

    domain_special_characters_to_alpha_ratio = {}
    for domain, cnt in non_alphanumeric_characters_per_domain.items():
        domain_special_characters_to_alpha_ratio[domain] = float(np.mean(cnt))

    domain_average_words_per_document = {}
    for domain, cnt in words_per_doc_per_domain.items():
        domain_average_words_per_document[domain] = float(np.mean(cnt))

    domain_average_sentence_length = {}
    for domain, cnt in sentence_lengths_per_domain.items():
        domain_average_sentence_length[domain] = float(np.mean(cnt))

    domain_median_sentence_length = {}
    for domain, cnt in sentence_lengths_per_domain.items():
        domain_median_sentence_length[domain] = float(np.median(cnt))

    domain_std_sentence_length = {}
    for domain, cnt in sentence_lengths_per_domain.items():
        domain_std_sentence_length[domain] = float(np.std(cnt))

    domain_special_characters_to_alpha_ratio = {}
    for domain, cnt in non_alphanumeric_characters_per_domain.items():
        domain_special_characters_to_alpha_ratio[domain] = float(np.mean(cnt))

    domain_most_frequent_2_grams = {}
    for domain, cnt in domain_2_grams_counters.items():
        domain_most_frequent_2_grams[domain] = dict(cnt.most_common(20))

    domain_most_frequent_3_grams = {}
    for domain, cnt in domain_3_grams_counters.items():
        domain_most_frequent_3_grams[domain] = dict(cnt.most_common(20))

    domain_most_frequent_4_grams = {}
    for domain, cnt in domain_4_grams_counters.items():
        domain_most_frequent_4_grams[domain] = dict(cnt.most_common(20))

    domain_most_frequent_5_grams = {}
    for domain, cnt in domain_5_grams_counters.items():
        domain_most_frequent_5_grams[domain] = dict(cnt.most_common(20))

    domain_most_frequent_6_grams = {}
    for domain, cnt in domain_6_grams_counters.items():
        domain_most_frequent_6_grams[domain] = dict(cnt.most_common(20))

    return DatasetMetrics(
        number_of_docs=len(docs),
        domains_counter=dict(domains_counter.most_common(20)),
        dataset_most_common_pos=dict(pos_counter.most_common(10)),
        dataset_most_common_words=dict(word_counter.most_common(20)),
        dataset_least_common_words=dict(word_counter.most_common()[::-20]),
        dataset_total_words=total_words,
        dataset_average_words_per_document=avg_words_per_doc,
        dataset_average_sentence_length=avg_sentence_lengths,
        dataset_median_sentence_length=median_sentence_lengths,
        dataset_std_sentence_length=std_sentence_lengths,
        dataset_shannon_entropy=entropy,
        dataset_special_characters_to_alpha_ratio=float(np.mean(non_alphanumeric_characters)),
        dataset_most_frequent_2_grams=dict(all_2_grams_counter.most_common(20)),
        dataset_most_frequent_3_grams=dict(all_3_grams_counter.most_common(20)),
        dataset_most_frequent_4_grams=dict(all_4_grams_counter.most_common(20)),
        dataset_most_frequent_5_grams=dict(all_5_grams_counter.most_common(20)),
        dataset_most_frequent_6_grams=dict(all_6_grams_counter.most_common(20)),
        domain_most_common_pos=domain_most_common_pos,
        domain_most_common_words=domain_most_common_words,
        domain_least_common_words=domain_least_common_words,
        domain_total_words=total_words_per_domain,
        domain_average_words_per_document=domain_average_words_per_document,
        domain_average_sentence_length=domain_average_sentence_length,
        domain_median_sentence_length=domain_median_sentence_length,
        domain_std_sentence_length=domain_std_sentence_length,
        domain_shannon_entropy=entropy_per_domain,
        domain_special_characters_to_alpha_ratio=domain_special_characters_to_alpha_ratio,
        domain_most_frequent_2_grams=domain_most_frequent_2_grams,
        domain_most_frequent_3_grams=domain_most_frequent_3_grams,
        domain_most_frequent_4_grams=domain_most_frequent_4_grams,
        domain_most_frequent_5_grams=domain_most_frequent_5_grams,
        domain_most_frequent_6_grams=domain_most_frequent_6_grams,
        dataset_top_tfidf=dataset_top_tfidf,
        # domain_top_tfidf={d: {feature_names[i]: float(round(np.asarray(tfidf_matrix[idxs].mean(axis=0)).ravel()[i],4)) for i in np.argsort(np.asarray(tfidf_matrix[idxs].mean(axis=0)).ravel())[::-1][:20]} for d, idxs in {d: [i for i,doc in enumerate(docs) if doc.domain==d] for d in domains}.items()},
        dataset_styles=dataset_style_counter,
        dataset_target_audience=dataset_target_audience_counter,
        domain_styles={d: dict(domain_style_counter[d].most_common()) for d in domain_style_counter.keys()},
        domain_target_audience={d: dict(domain_target_audience_counter[d].most_common()) for d in domain_target_audience_counter.keys()},
    )
