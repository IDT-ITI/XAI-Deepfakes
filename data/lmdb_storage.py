import hashlib
import io
import logging
from pathlib import Path
from typing import Optional, Union

import click
import lmdb
import pandas as pd
import tqdm

__version__: str = "0.0.2-alpha"
__author__: str = (
    "Dimitrios Karageorgiou (adapted by Spiros Baxevanakis spirosbax@iti.gr)"
)
__email__: str = "dkarageo@iti.gr"


class LMDBFileStorage:
    """A file storage for handling large datasets based on LMDB."""

    def __init__(
        self,
        db_path: Path,
        map_size: int = 2 * 1024 * 1024 * 1024 * 1024,  # 2TB
        read_only: bool = False,
        max_readers: int = 128,
    ):
        self.db: lmdb.Environment = lmdb.open(
            str(db_path),
            map_size=map_size,
            subdir=False,
            readonly=read_only,
            max_readers=max_readers,
            lock=False,
            sync=False,
        )

    def open_file(self, file_id: str, mode: str = "r") -> io.BytesIO:
        """Returns a file-like stream of a file in the database."""

        with self.db.begin(buffers=True) as trans:
            data = trans.get(file_id.encode("ascii"))
        stream: io.BytesIO = io.BytesIO(data)  # type: ignore

        return stream

    def write_file(self, file_id: str, file_data: bytes) -> None:
        with self.db.begin(write=True) as trans:
            trans.put(file_id.encode("ascii"), file_data)

    def get_all_ids(
        self,
        max_keys: Union[int, None] = None,
        decode: bool = False,
        verbose: bool = True,
    ) -> list[str]:
        with self.db.begin() as trans:
            cursor = trans.cursor()
            ids = []
            if verbose:
                pbar = tqdm.tqdm(cursor, desc="Getting all ids", unit="key")
            else:
                pbar = cursor
            for key, _ in pbar:
                if decode:
                    ids.append(key.decode("ascii"))
                else:
                    ids.append(key)
                if max_keys is not None and len(ids) >= max_keys:
                    break
        return ids

    def close(self) -> None:
        self.db.close()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "-csv",
    "--csv_file",
    required=True,
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    help="Path to the csv file.",
)
# add boolean option for csv header existence
@click.option(
    "-h",
    "--header",
    is_flag=True,
    default=False,
    help="Boolean flag for csv header existence.",
)
# if header exists, add option for path and label column names
@click.option(
    "-pc",
    "--path_column",
    type=str,
    default=None,
    help="Name of the column containing the paths.",
)
@click.option(
    "-d",
    "--database",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the database. If the file does not "
    "exist, a new database is generated. Otherwise, it should point to a "
    "previous instance of the LMDB, where data will be added.",
)
@click.option(
    "-b",
    "--base_dir",
    required=False,
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    help="Base directory of the dataset. Paths inside the CSV should be relative "
    "to that path. When not provided, the directory of the CSV file is "
    "considered as the base directory.",
    default=None,
)
@click.option(
    "-ms",
    "--map_size",
    type=int,
    default=2 * 1024 * 1024 * 1024 * 1024,  # 2TB
    help="Map size for the database, in bytes.",
)
@click.option(
    "-bs",
    "--batch_size",
    type=int,
    default=1000,
    help="Batch size for writing to the database.",
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing files in the database.",
)
@click.option(
    "-src",
    "--src_db_path",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    default=None,
    help="Path to the source database. If provided, files will be copied from "
    "this database to the destination database.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print progress messages and warnings/erros.",
)
@click.option(
    "-ld",
    "--log_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=None,
)
def add_csv(
    csv_file: Path,
    header: bool,
    path_column: str,
    database: Path,
    base_dir: Optional[Path] = None,
    map_size: int = 2 * 1024 * 1024 * 1024 * 1024,  # 2TB
    batch_size: int = 1000,
    src_db_path: Optional[Path] = None,
    overwrite: bool = False,
    verbose: bool = False,
    log_dir: Union[Path, None] = None,
) -> None:
    db: LMDBFileStorage = LMDBFileStorage(database, map_size=map_size)
    
    if csv_file.exists():
        add_csv_to_db(
            csv_file,
            db,
            header,
            path_column,
            batch_size,
            overwrite,
            base_dir,
            src_db_path,
            verbose,
            log_dir,
        )
    else:
        if verbose:
            print("Could not find {csv_file}.")

    db.close()


@cli.command()
@click.option("-s", "--src", required=True,
              type=click.Path(dir_okay=False, path_type=Path, exists=True),
              help="Database whose files will be added to the destination database.")
@click.option("-d", "--dest", required=True,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Database where file from source database will be added.")
def add_db(
    src: Path,
    dest: Path
) -> None:
    """Adds all the contents of a database to another."""
    src_db: LMDBFileStorage = LMDBFileStorage(src, read_only=True)
    dest_db: LMDBFileStorage = LMDBFileStorage(dest)


    for k in tqdm.tqdm(src_db.get_all_ids(), desc="Copying files", unit="file"):
        k = str(k, 'UTF-8')
        src_file: io.BytesIO = src_db.open_file(k, mode="b")
        dest_db.write_file(k, src_file.read())


    src_db.close()
    dest_db.close()


@cli.command()
@click.option(
    "-csv",
    "--csv_file",
    required=True,
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    help="Path to the csv file.",
)
@click.option(
    "-b",
    "--base_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    help="Base directory of the dataset. Paths inside the CSV should be relative "
    "to that path. When not provided, the directory of the CSV file is "
    "considered as the base directory.",
)
# add boolean option for csv header existence
@click.option(
    "-h",
    "--header",
    is_flag=True,
    default=False,
    help="Boolean flag for csv header existence.",
)
# if header exists, add option for path and label column names
@click.option(
    "-pc",
    "--path_column",
    type=str,
    default=None,
    help="Name of the column containing the paths.",
)
@click.option(
    "-d",
    "--database",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the database.",
)
@click.option(
    "-ld",
    "--log_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=None,
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print progress messages and warnings/erros.",
)
def verify_csv(
    csv_file: Path,
    base_dir: Path,
    header: bool,
    path_column: str,
    database: Path,
    log_dir: Union[Path, None] = None,
    verbose: bool = False,
) -> None:
    db: LMDBFileStorage = LMDBFileStorage(database, read_only=True)
    if csv_file.exists():
        # set up logging
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{csv_file.stem}.log"
            logging.basicConfig(
                filename=log_file,
                filemode="w",
                format="%(asctime)s %(levelname)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO,
            )        

        verify_csv_in_db(csv_file, db, base_dir, header, path_column, verbose)
    else:
        if verbose:
            print(f'Could not find {csv_file}.')
            print(f"Please make sure that the csv exists and is named correctly.")
    db.close()


@cli.command()
@click.option(
    "-d",
    "--database",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
    help="Database to replace keys.",
)
@click.option(
    "-b",
    "--base_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    help="Base directory of the dataset. Paths inside the CSV should be relative "
    "to that path. When not provided, the directory of the CSV file is "
    "considered as the base directory.",
)
# add boolean option for csv header existence
@click.option(
    "-h",
    "--header",
    is_flag=True,
    default=False,
    help="Boolean flag for csv header existence.",
)
# if header exists, add option for path and label column names
@click.option(
    "-pc",
    "--path_column",
    type=str,
    default=None,
    help="Name of the column containing the paths.",
)
@click.option(
    "-bs",
    "--batch_size",
    type=int,
    default=1000,
    help="Batch size for writing to the database.",
)
@click.option(
    "-csvr",
    "--csv_to_remove",
    required=True,
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    help="Path to the CSV file that contains the keys to remove.",
)
@click.option(
    "-csva",
    "--csv_to_add",
    required=True,
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
    help="Path to the CSV file that contains the keys to add.",
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing files in the database.",
)
def replace_keys(
    database: Path,
    base_dir: Path,
    header: bool,
    path_column: str,
    batch_size: int,
    csv_to_remove: Path,
    csv_to_add: Path,
    overwrite: bool,
    verbose: bool = True,
) -> None:
    """Replaces keys in a database with keys from a CSV file."""
    db: LMDBFileStorage = LMDBFileStorage(database, read_only=False)

    if verbose:
        logging.info(f"REMOVING CSV: {str(csv_to_remove)}")

    delete_csv_from_db(csv_to_remove, db, header, path_column, batch_size, verbose)
    add_csv_to_db(
        csv_to_add, db, base_dir, header, path_column, batch_size, overwrite, verbose
    )

    db.close()


@cli.command()
@click.option(
    "-d",
    "--database",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
    help="Database whose keys will be printed.",
)
@click.option(
    "-m",
    "--max_keys",
    required=False,
    default=None,
    type=int,
    help="Maximum number of keys to be printed.",
)
@click.option(
    "-o",
    "--output_file",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the output file where the keys will be written. ",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print progress messages and warnings/erros.",
)
def list_db(
    database: Path,
    max_keys: Union[int, None],
    output_file: Union[Path, None],
    verbose: bool = False,
) -> None:
    """Lists the contents of a file storage."""
    db: LMDBFileStorage = LMDBFileStorage(database, read_only=True)
    keys = db.get_all_ids(max_keys=max_keys, decode=True, verbose=verbose)
    if output_file is not None:
        with open(output_file, "w") as f:
            for item in keys:
                f.write("%s\n" % item)


@cli.command()
@click.option(
    "-d",
    "--database",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
    help="Database whose stats will be printed.",
)
def print_stats(database: Path) -> None:
    """Prints the stats of a file storage."""
    db: LMDBFileStorage = LMDBFileStorage(database, read_only=True)
    stats = db.db.stat()
    print(stats)


def add_csv_to_db(
    csv_file: Path,
    db: LMDBFileStorage,
    header: bool,
    path_column: str,
    batch_size: int,
    overwrite: bool,
    base_dir: Optional[Path] = None,
    src_db_path: Optional[Path] = None,
    verbose: bool = True,
    log_dir: Union[Path, None] = None,
) -> int:
    """Adds the contents of the file paths included in a CSV file into an LMDB File Storage.

    Paths of the files, relative to the base dir, are utilized as keys into the storage.
    Thus, the maximum allowed path length is 511 bytes.

    In that case, keys represent the file structure relative to the base dir.

    :param csv_file: Path to a CSV file describing a dataset.
    :param db: An instance of LMDB File Storage, where files will be added.
    :param base_dir: Directory where paths included into the CSV file are relative to.
    :param header: Boolean flag for csv header existence.
    :param path_column: Name of the column containing the paths.
    :param batch_size: Number of files to be written in a single transaction.
    :param overwrite: When set to True, existing files in the database will be overwritten.
    :param verbose: When set to False, progress messages will not be printed.
    """
    entries: list[str] = read_csv_file(csv_file, header, path_column, verbose)

    # set up logging
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{csv_file.stem}.log"
        logging.basicConfig(
            filename=log_file,
            filemode="w",
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    if verbose:
        logging.info(f"ADDING CSV: {str(csv_file)}")

    src_db: Optional[LMDBFileStorage] = None
    if src_db_path is not None and Path(src_db_path).exists() and base_dir is None:
        logging.info(f"SOURCE DATABASE: {str(src_db_path)}")
        src_db: LMDBFileStorage = LMDBFileStorage(src_db_path, read_only=True)

    files_written = write_files_to_db(
        entries, db, batch_size, base_dir, src_db, overwrite, verbose
    )

    if verbose:
        logging.info(f"FILES WRITTEN: {files_written}")

    return files_written


def delete_csv_from_db(
    csv_file: Path,
    db: LMDBFileStorage,
    header: bool,
    path_column: str,
    batch_size: int,
    verbose: bool = True,
) -> int:
    entries: list[str] = read_csv_file(csv_file, header, path_column, verbose)

    if verbose:
        logging.info(f"DELETING CSV: {str(csv_file)}")

    n_deleted_files = delete_files_from_db(entries, db, batch_size, verbose)

    if verbose:
        logging.info(f"FILES DELETED: {n_deleted_files}")

    return n_deleted_files


def delete_files_from_db(
    files: list[str],
    db: LMDBFileStorage,
    batch_size: int,
    verbose: bool = True,
) -> int:
    n_files = len(files)
    n_deleted_files = 0

    if verbose:
        pbar = tqdm.tqdm(files, desc="Deleting CSV data from database", unit="file")
    else:
        pbar = files

    trans = db.db.begin(write=True)
    try:
        for i, path in enumerate(pbar):
            key = str(path).encode("ascii")

            if trans.get(key) is not None:
                deleted = trans.delete(key)
                if deleted:
                    n_deleted_files += 1

            # Commit the transaction and begin a new one after every batch_size entries
            if (i + 1) % batch_size == 0:
                trans.commit()
                trans = db.db.begin(write=True)

        # Commit the transaction for the last batch of entries, if any
        if n_files % batch_size != 0:
            trans.commit()

    except Exception as e:
        if trans:
            trans.abort()
        print(f"An error occurred: {e}")
        raise

    return n_deleted_files


def verify_csv_in_db(
    csv_file: Path,
    db: LMDBFileStorage,
    base_dir: Path,
    header: bool,
    path_column: str,
    verbose: bool = True,
) -> int:
    entries: list[str] = read_csv_file(csv_file, header, path_column, verbose)
    files_verified = verify_files_in_db(entries, base_dir, db, verbose)

    if verbose:
        logging.info(f"FILES VERIFIED: {files_verified}")

    return files_verified


def write_files_to_db(
    files: list[str],
    db: LMDBFileStorage,
    batch_size: int,
    base_dir: Optional[Path] = None,
    src_db: Optional[LMDBFileStorage] = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> int:
    n_files = len(files)

    pbar = tqdm.tqdm(files, desc="Writing CSV data to database", unit="file") if verbose else files

    trans = db.db.begin(write=True)

    try:
        for i, path in enumerate(pbar):
            try:
                if base_dir is not None:
                    data: bytes = read_raw_file(base_dir / path)
                elif src_db is not None:
                    src_file: io.BytesIO = src_db.open_file(path, mode="b")
                    data: bytes = src_file.read()
            except Exception as e:
                logging.error(f"Error: {e}")
                continue

            key = str(path).encode("ascii")
            if trans.get(key) is None or overwrite:
                trans.put(key, data)

            if (i + 1) % batch_size == 0:
                trans.commit()
                trans = db.db.begin(write=True)

        if n_files % batch_size != 0:
            trans.commit()

    except Exception as e:
        if trans:
            trans.abort()
        src_db.close()
        logging.error(f"An error occurred: {e}")
        raise

    return n_files

def verify_files_in_db(
    files: list[str], base_dir: Path, db: LMDBFileStorage, verbose: bool = True
) -> int:

    if verbose:
        pbar = tqdm.tqdm(files, desc="Verifying CSV data in database", unit="file")
    else:
        pbar = files

    verified: int = 0
    for p in pbar:
        attempts = 0
        while attempts < 10:
            try:
                # Calculate md5 hash of the file in csv.
                with (base_dir / p).open("rb") as f:
                    csv_file_hash: str = md5(f)
                break
            except OSError as e:
                attempts += 1
                logging.error(f"Attempt {attempts}: Error reading file {p}. Retrying...")
                if attempts >= 10:
                    logging.error(f"Failed to read file {p} after 10 attempts. Error: {e}")
                    break  # Exit the loop and continue with the next file

        if attempts < 10:
            # Calculate md5 hash of the file in db if the file was successfully read.
            db_file: io.BytesIO = db.open_file(p, mode="b")
            db_file_hash: str = md5(db_file)
            if csv_file_hash == db_file_hash:
                verified += 1
            else:
                logging.error(f"File in DB not matching file in CSV: {str(p)}")
        else:
            logging.error(f"Skipped file due to repeated read errors: {str(p)}")

    return verified


def read_csv_file(
    csv_file: Path, header: bool, path_column: str, verbose: bool = True
) -> list[str]:
    # Read the whole csv file.
    if verbose:
        logging.info(f"READING CSV: {str(csv_file)}")

    # check if file exists
    if not csv_file.exists():
        raise FileNotFoundError(f"Could not find {str(csv_file)}")

    # TODO handle case where there is header
    df = pd.read_csv(csv_file, header="infer" if header else None, sep=" ")
    if path_column is not None:
        entries: list[str] = df[path_column].tolist()
    else:
        entries: list[str] = df.iloc[:, 0].tolist()

    if verbose:
        logging.info(f"TOTAL ENTRIES: {len(entries)}")

    return entries


def read_raw_file(p: Path) -> bytes:
    attempts = 0
    while attempts < 10:
        try:
            with p.open("rb") as f:
                data: bytes = f.read()
            return data
        except OSError as e:
            attempts += 1
            logging.info(f"Attempt {attempts}: Error reading file {p}. Retrying...")

    # If all attempts fail, handle or raise the exception
    raise OSError(f"Failed to read file {p} after 10 attempts.")


def md5(stream) -> str:
    """Calculates md5 hash of a file-like stream."""
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: stream.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    cli()
