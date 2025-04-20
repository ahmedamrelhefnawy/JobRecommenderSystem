import numpy as np
from typing import List, Tuple, Optional

def rename_key(dictionary, old_key, new_key):
    if old_key in dictionary:
        dictionary[new_key] = dictionary.pop(old_key)

def normalize(array: np.array):
    '''
    Normalize a numpy array to the range [0, 1].
    '''
    if array.min() == 0 and array.max() == 1:
        return array
    elif array.min() != array.max(): 
        return (array - array.min()) / (array.max() - array.min())
    else:
        return np.zeros_like(array)

def filter_recommendations(recommendations: List[Tuple[int, float]], max_recommendations: int, threshold: Optional[float] = None) -> List[int]:
    """Filter job recommendations based on score threshold and maximum count.
    
    Args:
        recommendations: List of tuples containing (job_id, score)
        max_recommendations: Maximum number of recommendations to return
        threshold: Minimum score threshold (optional)
    
    Returns:
        List of filtered job IDs
    """
    if not recommendations:
        return []
        
    if threshold:
        # Filter recommendations by threshold
        recommendations = [job_id for job_id, score in recommendations if score >= threshold]
    else:
        # Filter recommendations by max_recommendations
        limit = min(max_recommendations, len(recommendations))
        recommendations = [job_id for job_id, _ in recommendations[:limit]]
    
    return recommendations

