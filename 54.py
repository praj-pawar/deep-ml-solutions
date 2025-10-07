import numpy as np

def rnn_forward(input_sequence: list[list[float]], 
                initial_hidden_state: list[float], 
                Wx: list[list[float]], 
                Wh: list[list[float]], 
                b: list[float]) -> list[float]:
    
    # Convert everything to numpy arrays
    input_seq=np.array(input_sequence,dtype=float)
    hs=np.array(initial_hidden_state,dtype=float)
    wx=np.array(Wx,dtype=float)
    wh=np.array(Wh,dtype=float)
    b=np.array(b,dtype=float)

    for x in input_seq:
        hs=np.tanh(wx @ x + wh @ hs +b)
    
    return np.round(hs,4).tolist()
   

