def area_triangle(base, height):
    """
    Calculates triangle area with given base and height length.
    Input: base (float), height (float)
    Output: area_out (float )
    """
    if base < 0 and height < 0:
        raise ValueError("The base and hight length must be >0")
    elif base < 0:
        raise ValueError("The base length must be >0")
    elif height < 0:
        raise ValueError("The height length must be >0")

    area_out = 0.5 * base * height
    print("The triangle area is {:4.2f}cm2.".format(area_out))
    return area_out
