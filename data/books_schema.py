# retrieval/data/schema.py
from .booksdataset import DEFAULT_RATINGS_COLUMN_NAMES
UID_KEY, ITEMID_KEY, RATING_KEY = DEFAULT_RATINGS_COLUMN_NAMES[:3]

DEFAULT_RATINGS_COLUMN_NAMES: List[str] = ['User-ID', 'ISBN', 'Book-Rating']
DEFAULT_BOOKS_COLUMN_NAMES: List[str] = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
DEFAULT_USERS_COLUMN_NAMES: List[str] = ['User-ID', 'Location', 'Age']

USER_COUNTRY = "Location"
USER_AGE     = "Age"
ITEM_TITLE   = "Book-Title"
ITEM_AUTHOR  = "Book-Author"
ITEM_YEAR    = "Year-Of-Publication"
ITEM_PUB     = "Publisher"
