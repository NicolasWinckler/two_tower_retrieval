# bookcrossing.py
import os, re, hashlib
from typing import Any, Callable, Dict, List, Optional, Union
from torch.utils.data import IterDataPipe
from torch.utils.data import IterableDataset
from torchrec.datasets.utils import LoadFiles, ReadLinesFromCSV, safe_cast

RATINGS_FILENAME = "Ratings.csv" # ['User-ID', 'ISBN', 'Book-Rating']
USERS_FILENAME   = "Users.csv" # ['User-ID', 'Location', 'Age']
BOOKS_FILENAME   = "Books.csv" # ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']

DEFAULT_RATINGS_COLUMN_NAMES: List[str] = ['User-ID', 'ISBN', 'Book-Rating']
DEFAULT_BOOKS_COLUMN_NAMES: List[str] = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
DEFAULT_USERS_COLUMN_NAMES: List[str] = ['User-ID', 'Location', 'Age']

# Output column order (like movielens.py does)
# DEFAULT_COLUMN_NAMES: List[str] = [
#     "userId", "bookId", "rating",  # core
#     "country", "age",              # from Users.csv (optional)
#     "title", "author", "year", "publisher"  # from Books.csv (optional)
# ]

DEFAULT_COLUMN_NAMES: List[str] = (
    DEFAULT_RATINGS_COLUMN_NAMES  + DEFAULT_USERS_COLUMN_NAMES[1:] + DEFAULT_BOOKS_COLUMN_NAMES[1:5]
)

# Simple casters
COLUMN_TYPE_CASTERS: List[Callable[[Union[float,int,str]], Union[float,int,str]]] = [
    lambda v: safe_cast(v, str, ""),     # userId 
    lambda v: safe_cast(v, str, ""),     # bookId (ISBN)
    lambda v: safe_cast(v, float, 0.0),  # rating
    lambda v: safe_cast(v, str, ""),     # country
    lambda v: safe_cast(v, int, 0),      # age
    lambda v: safe_cast(v, str, ""),     # title
    lambda v: safe_cast(v, str, ""),     # author
    lambda v: safe_cast(v, int, 0),      # year
    lambda v: safe_cast(v, str, ""),     # publisher
]

def _default_row_mapper(example: List[str]) -> Dict[str, Union[float,int,str]]:
    # Map to dict with the types we want
    return {
        DEFAULT_COLUMN_NAMES[i]: COLUMN_TYPE_CASTERS[i](val)
        for i, val in enumerate(example)
    }

def _parse_country(loc: str) -> str:
    if not isinstance(loc, str): return ""
    parts = [p.strip() for p in loc.split(",") if p.strip()]
    return parts[-1] if parts else ""

def _join_with_users(datapipe: IterDataPipe, root: str, , filter_to_catalog: bool = False) -> IterDataPipe:
    users_path = os.path.join(root, USERS_FILENAME)
    udp = LoadFiles((users_path,), mode="r")
    udp = ReadLinesFromCSV(udp, skip_first_line=True, delimiter=",")
    # Build a small in‑memory map: User‑ID -> (country, age)
    user_map: Dict[str, List[str]] = {}
    for row in udp:
        uid, location, age = row[0], row[1], row[2] if len(row) > 2 else ""
        user_map[uid] = [_parse_country(location), age]

    if filter_to_catalog:
        # Drop ratings rows whose user is not in Users.csv
        datapipe = datapipe.filter(lambda r: r[0] in user_map)

    def join_user(row: List[str]) -> List[str]:
        # here row from datapipe
        # ratings row is [User-ID, ISBN, Book-Rating]
        uid = row[0]
        country, age = user_map.get(uid, ["", "0"])
        return row + [country, age]

    return datapipe.map(join_user)

def _join_with_books(datapipe: IterDataPipe, root: str, , filter_to_catalog: bool = False) -> IterDataPipe:
    books_path = os.path.join(root, BOOKS_FILENAME)
    bdp = LoadFiles((books_path,), mode="r")
    bdp = ReadLinesFromCSV(bdp, skip_first_line=True, delimiter=",")
    # Map: ISBN -> [title, author, year, publisher]
    book_map: Dict[str, List[str]] = {}
    for row in bdp:
        isbn = row[0]
        title = row[1] if len(row) > 1 else ""
        author = row[2] if len(row) > 2 else ""
        # year can be messy; keep as string for now
        year = row[3] if len(row) > 3 else "0"
        publisher = row[4] if len(row) > 4 else ""
        book_map[isbn] = [title, author, year, publisher]

    if filter_to_catalog:
        # Drop ratings rows whose ISBN is not in Books.csv
        datapipe = datapipe.filter(lambda r: r[1] in book_map)

    def join_book(row: List[str]) -> List[str]:
        # current row: [User-ID, ISBN, Book-Rating, country, age]
        isbn = row[1]
        title, author, year, publisher = book_map.get(isbn, ["", "", "0", ""])
        return row + [title, author, year, publisher]

    return datapipe.map(join_book)

def _bookcrossing(
    root: str,
    *,
    include_users_data: bool = True,
    include_books_data: bool = True,
    filter_to_catalog: bool = False,
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    **open_kw,
) -> IterDataPipe:
    # Start from Ratings.csv
    ratings_path = os.path.join(root, RATINGS_FILENAME)
    dp = LoadFiles((ratings_path,), mode="r", **open_kw)
    dp = ReadLinesFromCSV(dp, skip_first_line=True, delimiter=",")

    # Ratings.csv columns: ['User-ID', 'ISBN', 'Book-Rating']
    # Reorder/augment to match DEFAULT_COLUMN_NAMES
    # First, rename to [userId, bookId, rating]
    dp = dp.map(lambda r: [r[0], r[1], r[2]])

    if include_users_data:
        dp = _join_with_users(dp, root, filter_to_catalog=filter_to_catalog)   # adds [country, age]
    else:
        dp = dp.map(lambda r: r + ["", "0"])

    if include_books_data:
        dp = _join_with_books(dp, root, filter_to_catalog=filter_to_catalog)   # adds [title, author, year, publisher]
    else:
        dp = dp.map(lambda r: r + ["", "", "0", ""])

    if row_mapper:
        dp = dp.map(row_mapper)
    return dp

def bookcrossing(
    root: str,
    *,
    include_users_data: bool = True,
    include_books_data: bool = True, 
    filter_to_catalog: bool = False,
    row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    **open_kw,
) -> IterDataPipe:
    """
    Book-Crossing style dataset loader.
    Args:
      root: directory containing Ratings.csv, Users.csv, Books.csv
      include_users_data / include_books_data: join side data
      row_mapper: maps a CSV row -> dict with keys DEFAULT_COLUMN_NAMES
    """
    return _bookcrossing(
        root,
        include_users_data=include_users_data,
        include_books_data=include_books_data,
        filter_to_catalog=filter_to_catalog,
        row_mapper=row_mapper,
        **open_kw,
    )


class BookCrossingIterableDataset(IterableDataset):
    """
    Wraps the bookcrossing() DataPipe so it can be used with torch.utils.data.DataLoader.
    Yields dicts with keys == DEFAULT_COLUMN_NAMES from your row_mapper.
    """
    def __init__(
        self,
        root: str,
        *,
        include_users_data: bool = True,
        include_books_data: bool = True,
        filter_to_catalog: bool = False,
        row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
    ) -> None:
        super().__init__()
        self._dp = bookcrossing(
            root,
            include_users_data=include_users_data,
            include_books_data=include_books_data,
            filter_to_catalog=filter_to_catalog,
            row_mapper=row_mapper,
        )

    def __iter__(self):
        return iter(self._dp)