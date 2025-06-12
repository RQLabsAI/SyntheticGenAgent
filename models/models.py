from typing import List, Dict

from pydantic import BaseModel


class Token(BaseModel):
    token: str
    pos: str


class Document(BaseModel):
    text: str
    domain: str
    tokens: List[Token]
    sentences: List[List[Token]]
    pisarek_index: float
    style: str
    target_audience: str


class DatasetMetrics(BaseModel):
    number_of_docs: int
    domains_counter: dict
    dataset_most_common_pos: Dict[str, int]
    dataset_most_common_words: Dict[str, int]
    dataset_least_common_words: Dict[str, int]
    dataset_total_words: int
    dataset_average_words_per_document: float
    dataset_average_sentence_length: float
    dataset_median_sentence_length: float
    dataset_std_sentence_length: float
    dataset_shannon_entropy: float
    dataset_special_characters_to_alpha_ratio: float

    dataset_most_frequent_2_grams: Dict[str, int]
    dataset_most_frequent_3_grams: Dict[str, int]
    dataset_most_frequent_4_grams: Dict[str, int]
    dataset_most_frequent_5_grams: Dict[str, int]
    dataset_most_frequent_6_grams: Dict[str, int]

    domain_most_common_pos: Dict[str, Dict[str, int]]
    domain_most_common_words: Dict[str, Dict[str, int]]
    domain_least_common_words: Dict[str, Dict[str, int]]
    domain_total_words: Dict[str, int]
    domain_average_words_per_document: Dict[str, float]
    domain_average_sentence_length: Dict[str, float]
    domain_median_sentence_length: Dict[str, float]
    domain_std_sentence_length: Dict[str, float]
    domain_shannon_entropy: Dict[str, float]
    domain_special_characters_to_alpha_ratio: Dict[str, float]

    domain_most_frequent_2_grams: Dict[str, Dict[str, int]]
    domain_most_frequent_3_grams: Dict[str, Dict[str, int]]
    domain_most_frequent_4_grams: Dict[str, Dict[str, int]]
    domain_most_frequent_5_grams: Dict[str, Dict[str, int]]
    domain_most_frequent_6_grams: Dict[str, Dict[str, int]]

    dataset_top_tfidf: Dict[str, float]

    dataset_styles: Dict[str, int]
    dataset_target_audience: Dict[str, int]

    domain_styles: Dict[str, Dict[str, int]]
    domain_target_audience: Dict[str, Dict[str, int]]