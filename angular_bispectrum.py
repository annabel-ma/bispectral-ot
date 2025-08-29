import torch
import math

class Polar:
    def __init__(self, N):
        self.N = N #discretization of angles

    def makePolar(self, img):
        if len(img.shape) != 3:
            if len(img.shape) == 2:
                img = torch.unsqueeze(img, 0)
            else: 
                raise ValueError(
                    f"[Polar] Unsupported image shape {img.shape}. "
                    "Expected (C, H, W) or (H, W) with C = 1 or 3."
                )

        H = img.shape[1]
        W = img.shape[2]
        R = min(W,H) // 2
        
        def value(r, theta):
            new_x = int(W / 2 + r * math.cos(theta))
            new_y = int(H / 2 + r * math.sin(theta))

            # clamp indices (oob set to 0)
            if new_x < 0 or new_x >= W or new_y < 0 or new_y >= H:
                return torch.zeros(3) 
            
            return img[:, new_y, new_x]

        def values(r):
            return [(r ** 1) * value(r, i * 2 * math.pi / self.N) for i in range(self.N)]

        # build polar
        polar_list = []
        for r in range(R):
            polar_list.append(torch.stack(values(r), dim=0))

        polar_tensor = torch.stack(polar_list, dim=0)
        polar_tensor = polar_tensor.permute(2, 0, 1)
        return polar_tensor

# angular bispectrum class 
class PolarBispec:
    def __init__(self, N):
        self.N = N #discretization of angles
        self.polar = Polar(N)
        #now we define the kernel for fourier transform
        real, imag = torch.zeros(N,N), torch.zeros(N,N)
        for k in range(N):
            for g in range(N):
                real[k,g] = math.cos(2*math.pi*(k*g/N))
                imag[k,g] = -math.sin(2*math.pi*(k*g/N))
        self.fou_r = real
        self.fou_i = imag
        #now define the kernel to do the f_{i+j}^T
        self.four = torch.complex(self.fou_r, self.fou_i)
        self.conj = (self.four[None,:,:]*self.four[:,None,:]).conj()
        self.conj_r = self.conj.real
        self.conj_i = self.conj.imag
    
    #calculate bispectrum, (.,W,H) goes in and returns (.,R,N,N)
    def bispec(self, img):
        #computes one bispectrum per channel
        if len(img.shape) != 3:
            if len(img.shape) == 2:
                img = torch.unsqueeze(img, 0)
            else: 
                raise ValueError(
                    f"[Polar] Unsupported image shape {img.shape}. "
                    "Expected (C, H, W) or (H, W) with C = 1 or 3."
                )

        pol_img = self.polar.makePolar(img)

        rel  = torch.einsum("crg,kg->crk",  pol_img, self.fou_r)
        imag = torch.einsum("crg,kg->crk",  pol_img, self.fou_i)
        four_trans = torch.complex(rel, imag)  
        outer_pro = four_trans[:, :, :, None] * four_trans[:, :, None, :]

        r = torch.einsum("crg,ijg->crij", pol_img, self.conj_r)
        i = torch.einsum("crg,ijg->crij", pol_img, self.conj_i)
        conj_term = torch.complex(r, i)  

        bisp = outer_pro * conj_term  
        norm = torch.linalg.vector_norm(bisp, dim=(1,2,3), keepdim=True)
        bisp = torch.where(norm > 0, bisp / norm, bisp)
        return bisp 
