from typing import List, Callable, Tuple

import spacy
from pydantic import BaseModel


class Podsumowanie(BaseModel):
    podsumowanie: str
    ocena_jakosci: str

class MapReduce:
    def __init__(
            self,
            context_window: int,
            map_prompt: str,
            collapse_prompt: str,
            reduce_prompt: str,
            task: str,
            llm_call_map: Callable,
            llm_call_collapse: Callable,
            llm_call_reduce: Callable,
    ):
        """
        Implementacja MapReduce do przetworzenia w zasadzie dowolnego ZADANIA na dowolnej wielkości zbioru danych
        Bardzo się inspirowałem: https://arxiv.org/pdf/2410.09342 oraz https://github.com/thunlp/LLMxMapReduce?tab=readme-ov-file
        """
        self.context_window = context_window
        self.map_prompt = map_prompt
        self.collapse_prompt = collapse_prompt
        self.reduce_prompt = reduce_prompt
        self.task = task
        self.llm_call_map = llm_call_map
        self.llm_call_collapse = llm_call_collapse
        self.llm_call_reduce = llm_call_reduce

        self.nlp = spacy.load("pl_core_news_sm")

    def _chunk(self, texts: List[str]):
        """
        Funkcja do budowania chunków bazując na tokenach z modelów spacy.
        Zrobiona tak, by ucinała na całym zdaniu.
        """
        chunks = []
        current_chunk = []

        for text in texts:
            current_chunk = []
            current_chunk_len = 0

            doc = self.nlp(text)

            for sentence in doc.sents:
                word_count = sum(1 for token in sentence if not token.is_punct and not token.is_space)

                if current_chunk_len + word_count <= self.context_window:
                    # jeżeli mieści się w zdefiniowaną wielkość kontekstu
                    current_chunk.append(str(sentence))
                    current_chunk_len += word_count
                else:
                    # jeżeli nie mieści się w zdefiniowanej wielkości kontekstu
                    chunks.append("\n".join(map(lambda x: x.strip(), current_chunk)).strip())
                    current_chunk = [str(sentence)]
                    current_chunk_len = word_count

        if len(current_chunk) > 0:
            chunks.append("\n".join(map(lambda x: x.strip(), current_chunk)).strip())

        return chunks

    def map(self, texts: List[str]) -> Tuple[List[str], int, int]:
        """
        Funkcja mapowania. Jest to pierwszy etap. Jej zadaniem jest wygenerować wyniki wykonując "task" na każdym chunku.

        :param texts: teksty wejściowe
        :return: wyniki przetwarzania, ilość tokenów wejsciowych, ilość tokenów wyjściowych
        """
        chunks = self._chunk(texts)

        results = []
        total_in_tokens = 0
        total_out_tokens = 0
        for chunk in chunks:
            result, in_tokens, out_tokens = self.llm_call_map(self.map_prompt.replace("{chunk}", chunk).replace("{task}", self.task))
            results.append(result)
            total_in_tokens += in_tokens
            total_out_tokens += out_tokens
        return results, total_in_tokens, total_out_tokens

    @staticmethod
    def filter_out_samples(texts: List[str]) -> List[str]:
        """
        Prosta funkcja do filtrowania wyników. W prompcie Map powinniśmy napisać, że jeżeli model nie może zastosować wybranego "ZADANIA" do
        podanego chunku, powinien zwrócić [BRAK INFORMACJI]. Takie próbki na etapie collapse/reduce można odfiltrować.

        :param texts: wyniki przetwarzania z etapu Map
        :return: odfiltrowane próbki
        """
        return list(filter(lambda text: "[BRAK INFORMACJI]" not in text, texts))

    @staticmethod
    def build_wyniki(processed_outputs: List[str]) -> List[str]:
        """
        Przygotuj tekst wejściowy dla etapu Collapse/Reduce

        :param processed_outputs: lista wyników z etapu Map
        :return: lista tekstów
        """
        chunks_text = []
        for i, chunk in enumerate(processed_outputs):
            chunks_text.append(f""" --- WYNIK {i} --- \n {chunk} \n --- KONIEC WYNIKU {i} ---\n\n""")
        return chunks_text


    def collapse(self, processed_outputs: List[str]) -> Tuple[List[str], int, int]:
        """
        Etap collapse - zagreguj trochę wyniki Map, by do Collapse mogły wejść wszystkie na raz

        :param processed_outputs: przetworzone wyniki Map
        :return: lista tekstów po przetworzeniu
        """
        output_tokens = []

        # Wywal teksty, które nie mają wymaganych informacji
        processed_outputs = MapReduce.filter_out_samples(processed_outputs)

        # podlicz ilości tokenów z wynikami z Map
        for output in processed_outputs:
            doc = self.nlp(output)
            output_tokens.append(sum(1 for token in doc if not token.is_punct and not token.is_space))

        # Jeżeli suma outputu jest mniejsza niż context window, to zwróć, to co dostałeś
        if sum(output_tokens) < self.context_window:
            return processed_outputs, 0, 0
        else:
            # w przeciwnym wypadku zbuduj chunki i je przetwórz
            chunks = []

            current_chunk = []
            current_chunk_len = 0
            for part, part_size in zip(processed_outputs, output_tokens):
                # wypełnij context window
                if current_chunk_len + part_size <= self.context_window:
                    current_chunk.append(part)
                    current_chunk_len += part_size
                else:
                    chunks.append("\n".join(MapReduce.build_wyniki(current_chunk)))
                    current_chunk = [part]
                    current_chunk_len = part_size
            if len(current_chunk) > 0:
                chunks.append("\n".join(MapReduce.build_wyniki(current_chunk)))

            processed_chunks = []
            total_in_tokens = 0
            total_out_tokens = 0
            for chunk in chunks:
                prompt = self.collapse_prompt.replace("{results}", chunk).replace("{task}", self.task)
                processed_result, in_tokens, out_tokens = self.llm_call_collapse(prompt)
                processed_chunks.append(processed_result)
                total_in_tokens += in_tokens
                total_out_tokens += out_tokens

            return processed_chunks, total_in_tokens, total_out_tokens

    def reduce(self, processed_outputs: List[str]) -> Tuple[str, int, int]:
        """
        Etap reduce - ma za zadanie ogarnąć jedną uwspólnioną odpowiedź na podane zadanie

        :param processed_outputs: wyniki pochodzące z etapu Collapse/Map
        :return: finalna odpowiedź
        """
        results = "\n".join(MapReduce.build_wyniki(processed_outputs))

        prompt = self.reduce_prompt.replace("{results}", results).replace("{task}", self.task)

        return self.llm_call_reduce(prompt)

    def mapreduce(self, texts: List[str]) -> Tuple[str, int, int]:
        """
        Jedna funkcja, która dokonuje całego procesu MapReduce

        :param texts: teksty do przetworzenia
        :return: przetworzone wyniki
        """
        map_results, map_in_tokens, map_out_tokens = self.map(texts)
        collapsed_results, collapsed_in_tokens, collapsed_out_tokens = self.collapse(map_results)
        reduced_results, reduced_in_tokens, reduced_out_tokens = self.reduce(collapsed_results)
        return reduced_results, (map_in_tokens + collapsed_in_tokens + reduced_in_tokens), (map_in_tokens + collapsed_out_tokens + reduced_out_tokens)
