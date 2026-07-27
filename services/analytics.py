import re
from collections import Counter

# Set of common English words (stopwords) to ignore when finding key terms in the document
STOPWORDS = {
    "about", "after", "also", "among", "been", "before", "between", "could",
    "document", "each", "from", "have", "into", "more", "other", "page",
    "report", "shall", "should", "such", "than", "that", "their", "there",
    "these", "they", "this", "those", "through", "under", "were", "which",
    "while", "with", "would", "year", "your",
}

# Regular expression patterns for detecting specific financial/business themes
TOPIC_PATTERNS = {
    "Risk": r"\brisks?\b|\buncertaint\w*\b",
    "Growth": r"\bgrowth\b|\bexpansion\b",
    "Revenue": r"\brevenue\b|\bsales\b|\bturnover\b",
    "Profit": r"\bprofit\w*\b|\bearnings\b|\bmargin\w*\b",
    "Investment": r"\binvest\w*\b|\bcapital\b",
    "Market": r"\bmarkets?\b|\bindustry\b|\bsector\b",
    "Debt": r"\bdebt\b|\bborrow\w*\b|\bliabilit\w*\b",
    "Regulation": r"\bregulat\w*\b|\bcompliance\b|\bsebi\b",
}


def clean_text(text):
    """
    Normalizes a string's whitespace by replacing multiple spaces/tabs/newlines
    with a single space and stripping leading/trailing whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def extract_terms(text, limit=15):
    """
    Finds the most common meaningful words in the document.
    - Excludes stopwords.
    - Excludes short words (less than 4 letters).
    - Returns up to `limit` words with their count.
    """
    # Match words that start with a letter and contain letters/hyphens of length 4+
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{3,}\b", text.lower())
    
    # Count frequencies of words that are not in the STOPWORDS set
    counts = Counter(word for word in words if word not in STOPWORDS)
    
    # Return formatted list of top terms capitalized
    return [{"term": term.title(), "count": count} for term, count in counts.most_common(limit)]


def extract_financial_metrics(text, limit=30):
    """
    Extracts mentions of financial values, percentages, and financial years/quarters.
    Examples: $50 million, ₹10 crore, 15.3%, FY-24, Q3.
    """
    # Regex explanations:
    # 1. Currency: Symbol (₹, Rs, INR, $, USD, €, EUR) optionally followed by space, digits/commas,
    #    and optional words (crore, lakh, million, billion).
    # 2. Percentage: Numbers followed by %.
    # 3. Fiscal: FY or Q followed by numbers (e.g. FY-24, Q2).
    pattern = re.compile(
        r"(?:₹|Rs\.?|INR|\$|USD|€|EUR)\s?[\d,.]+(?:\s?(?:crore|lakh|million|billion))?"
        r"|(?:\d+(?:\.\d+)?)\s?%"
        r"|(?:FY|Q)[ -]?\d{2,4}(?:[ -]?\d{2})?",
        re.IGNORECASE,
    )
    seen = set()
    metrics = []
    
    # Iterate through all occurrences of the patterns in the document text
    for match in pattern.finditer(text):
        value = clean_text(match.group())
        key = value.lower()
        if key not in seen:
            seen.add(key)
            metrics.append(value)
        if len(metrics) >= limit:
            break
    return metrics


def build_analytics(chunks, page_count, image_count, filename):
    """
    Combines text from all chunks and calculates overall statistics of the document:
    - Words, characters, pages, images, and text chunks.
    - Frequencies of configured topics (Risk, Revenue, Growth, etc.).
    - Most common terms.
    - Financial metrics extracted.
    - Distribution of chunks across pages.
    """
    # Join all text segments to analyze the document as a whole
    combined = "\n".join(chunk["text"] for chunk in chunks)
    
    # Find all words in the combined text for word count
    words = re.findall(r"\b\w+\b", combined)
    
    # Count occurrences of the defined topic keyword patterns
    topics = [
        {"topic": name, "count": len(re.findall(pattern, combined, re.IGNORECASE))}
        for name, pattern in TOPIC_PATTERNS.items()
    ]
    # Sort topics in descending order of occurrence
    topics.sort(key=lambda item: item["count"], reverse=True)

    # Compute how many text chunks belong to each page
    page_distribution = Counter(chunk["page"] for chunk in chunks)
    
    return {
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks),
        "images": image_count,
        "words": len(words),
        "characters": len(combined),
        "top_terms": extract_terms(combined),
        "topics": topics,
        "financial_metrics": extract_financial_metrics(combined),
        "page_distribution": [
            {"page": page, "chunks": count}
            for page, count in sorted(page_distribution.items())
        ],
    }
