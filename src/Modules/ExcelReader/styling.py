from other_utils import convert_cell, make_list_linear
from typing import Union
from openpyxl.worksheet.cell_range import CellRange

def style_adjacent_cells(excel_file, start_cell: str, end_cell:str, *style_string: Union[str, tuple[str]]) -> None:
    """
    Styles all cells in a cell range with the same style.
    """
    start_cell, end_cell = convert_cell(cell_str=start_cell), convert_cell(cell_str=end_cell)
    range_of_cells = CellRange(range_string=f"{start_cell}:{end_cell}")


    style_string = make_list_linear(style_string)
    style_string = excel_file.add_style(*style_string)

    for cell in range_of_cells.cells:
        excel_file[cell[1], cell[0]].style = style_string

def style_cols(excel_file, col_str: Union[str, int, list[int, str], tuple[int, str]], *style_string: Union[str, tuple[str]], skip_init_rows = 0) -> None:
    """
    Styles all cells in a column range with the same style.
    """

    style_string = make_list_linear(style_string)
    style_string = excel_file.add_style(*style_string)

    rows_in_file = excel_file.rows
    while skip_init_rows > 0:
        next(rows_in_file)
        skip_init_rows -= 1

    if isinstance(col_str, str) or isinstance(col_str, int):
        if isinstance(col_str, str):
            col_str = convert_cell(col=col_str, return_type=int) - 1
        else:
            col = col_str - 1
        for row in rows_in_file:
            row[col].style = style_string
    else:
        cols = tuple(map(lambda x: convert_cell(col=x, return_type=int) - 1, col_str))
        for row in rows_in_file:
            for col in cols:
                row[col].style = style_string

def style_rows(excel_file, rows: Union[int, list[int], tuple[int]], *style_string: Union[str, tuple[str]], skip_init_rows = 0) -> None:
    """
    Styles all cells in a row range with the same style.
    """

    style_string = make_list_linear(style_string)
    style_string = excel_file.add_style(*style_string)

    if isinstance(rows, int):
        for col in range(1, excel_file.total_coloumns + 1):
            excel_file[col, rows].style = style_string
    else:
        rows_in_file = excel_file.rows
        curr_skipped = 0
        for to_skip in sorted(rows):
            while curr_skipped < to_skip:
                next(rows_in_file)
                curr_skipped += 1
            row = next(rows_in_file)
            curr_skipped += 1
            for index in range(skip_init_rows, excel_file.total_rows):
                row[index].style = style_string