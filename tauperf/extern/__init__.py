import rootpy.compiled as C
import os
HERE = os.path.dirname(os.path.abspath(__file__))
C.register_file(os.path.join(HERE, '1p_odd_BDT_1p_odd.class.C'),
                ['ReadBDT_1p_odd'])
C.register_file(os.path.join(HERE, 'mp_odd_BDT_mp_odd.class.C'),
                ['ReadBDT_mp_odd'])
C.register_file(os.path.join(HERE, '1p_even_BDT_1p_even.class.C'),
                ['ReadBDT_1p_even'])
C.register_file(os.path.join(HERE, 'mp_even_BDT_mp_even.class.C'),
                ['ReadBDT_mp_even'])

from rootpy.compiled import (
    ReadBDT_1p_odd, ReadBDT_1p_even, 
    ReadBDT_mp_odd, ReadBDT_mp_even)

__all__ = [
    'ReadBDT_1p_odd',
    'ReadBDT_mp_odd',
    'ReadBDT_1p_even',
    'ReadBDT_mp_even',
]
