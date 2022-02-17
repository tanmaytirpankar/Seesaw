import sqlite3


def db_setup(dbfile):
    """
    set up a sqlite3 database for writing results.

    Parameters
    ----------
    dbfile: string
        path to database to connect to. Will be created if does not exist.

    Returns
    -------
    connection, sqlcursor: pair
        a pair of database connection and sqlite3 cursor object for interacting with the database
    """
    connection = sqlite3.connect(dbfile);
    sqlcursor = connection.cursor()
    return connection, sqlcursor


def db_disconnect(connection, sqlcursor):
    """
    close database connections

    Parameters
    ----------
    connection:
        connection to the database to be closed
    sqlcursor:
        cursor to the database to be closed; must be from the same database as the connection param

    """
    assert sqlcursor.connection == connection, "unable to close database correctly"

    sqlcursor.close()
    connection.close()


def db_create_table(sqlcursor, table_name, defs):
    """
    Create a table using the given cursor

    Parameters
    ----------
    sqlcursor:
        cursor to a sqlite3 database in which to create table
    table_name: string
        name of table to create
    defs:
        col_name="DATA_TYPE" dict of all columns to include in this table
    """
    # Check if table exists
    sqlcursor.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='%s' ''' % table_name)
    if sqlcursor.fetchone()[0] == 1:
        print("table %s already exists" % table_name)
        exit(1)

    cols = ', '.join((' '.join((n.lower(), t.upper())) for n, t in defs.items()))
    sqlcursor.execute("CREATE TABLE " + table_name + " (" + cols + ")")


def table_insert(sqlcursor, table_name, data):
    """
    insert a record into the given table
    TODO: assertions to validate table cols against data

    Parameters
    ----------
    sqlcursor:
        cursor to a sqlite3 database in which table_name table exists.
    table_name: string
        name of table to insert data into
    data:
        dict of data to be inserted for a single row
    """
    cols, vals = zip(*data.items())
    sql_cmd = "INSERT INTO %s ('%s') VALUES (" % (table_name, "', '".join(cols))

    for v in vals:
        if isinstance(v, str):
            sql_cmd += "'" + v + "',"
        if isinstance(v, int):
            sql_cmd += str(v) + ","

    # remove extra comma
    sql_cmd = sql_cmd[:-1] + ")"
    sqlcursor.execute(sql_cmd)


def table_select(sqlcursor, table_name):
    """
    select all records in table

    Parameters
    ----------
    sqlcursor:
        cursor to a sqlite3 database in which table_name table exists.
    table_name: string
        name of table to select data from
    """
    sql_cmd = "SELECT * FROM %s" % table_name

    records = sqlcursor.execute(sql_cmd)
    for r in records:
        print(r)
