import statistics
from collections import Counter

from config import DECIMALS, POSITIVE_RATING_THRESHOLD
from state import RatingStats, SentimentSummary
from text_utils import words

PERCENT = 100
STAR_VALUES = range(1, 6)

POSITIVE_TERMS = frozenset(
    """
    excelente excelentes bueno buena buen buenos buenas perfecta perfecto delicioso deliciosa
    comodo comoda comodos ideal ligero ligeros facil rapido silencioso mejor recomendado
    recomendada recomendados parejo
    """.split()
)
NEGATIVE_TERMS = frozenset(
    """
    malo mala pesimo pesima roto rota atasca fragil pesado pesados ruidoso ruidosa raya
    desconectan decepcionante molesto dejaron nunca menos defectuoso
    """.split()
)


def rating_statistics(ratings: list[int]) -> RatingStats:
    distribution = Counter(ratings)
    if not ratings:
        return RatingStats(
            n=0, promedio=None, mediana=None, desviacion=None, porcentaje_positivas=None,
            distribucion={star: 0 for star in STAR_VALUES},
        )
    positive = sum(rating >= POSITIVE_RATING_THRESHOLD for rating in ratings)
    return RatingStats(
        n=len(ratings),
        promedio=round(statistics.fmean(ratings), DECIMALS),
        mediana=round(statistics.median(ratings), DECIMALS),
        desviacion=round(statistics.pstdev(ratings), DECIMALS),
        porcentaje_positivas=round(PERCENT * positive / len(ratings), DECIMALS),
        distribucion={star: distribution.get(star, 0) for star in STAR_VALUES},
    )


def polarity(text: str) -> float:
    tokens = words(text)
    positive = sum(token in POSITIVE_TERMS for token in tokens)
    negative = sum(token in NEGATIVE_TERMS for token in tokens)
    if positive + negative == 0:
        return 0.0
    return (positive - negative) / (positive + negative)


def review_sentiment(texts: list[str]) -> SentimentSummary:
    scores = [polarity(text) for text in texts]
    return SentimentSummary(
        n=len(scores),
        positivas=sum(score > 0 for score in scores),
        neutras=sum(score == 0 for score in scores),
        negativas=sum(score < 0 for score in scores),
        puntaje_medio=round(statistics.fmean(scores), DECIMALS) if scores else None,
    )
