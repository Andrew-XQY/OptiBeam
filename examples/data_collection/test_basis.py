from conf import *
from optibeam.basis import make_dct, make_hadamard, make_hg, make_walsh_hadamard

# -------------------- Example (remove or guard under __main__ as needed) --------------------
if __name__ == "__main__":
    # Quick sanity check / demo (small sizes to keep figure manageable)
    dim = 8
    
    temp = []
    temp.append(make_dct((dim, dim)))
    temp.append(make_hadamard(dim))
    temp.append(make_hg((dim, dim), extent=3.0))
    temp.append(make_walsh_hadamard(dim))  # just to test no error
    
    save_to = 'C:\\Users\\qiyuanxu\\Downloads\\'
    for i in range(len(temp)):
        print(temp[i].info())
        # Show and save example atlases
        temp[i].show_basis(save_dir=save_to)