"""Independent invariants for the new interfaces; run with unittest."""
import unittest
from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from fgplate import NavierFSDTPlate, al_al2o3, isotropic


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.p=NavierFSDTPlate(1.,1.4,.08,al_al2o3(2.3))

    def test_material_rejects_invalid_parameters(self):
        for name,values in {'n':[-1,np.nan,np.inf,True], 'nu':[-1,.5,np.nan],
                            'E_c':[0,-1,np.inf], 'rho_m':[0,np.nan]}.items():
            for v in values:
                with self.subTest(name=name,value=v),self.assertRaises(ValueError):
                    replace(al_al2o3(1),**{name:v})
        with self.assertRaises(ValueError): self.p.material.moment('typo',0,.1)
        with self.assertRaises(ValueError): self.p.material.moment('E',1.2,.1)
        with self.assertRaises(ValueError): self.p.material.E(.1,.1)

    def test_geometry_modes_and_load_validation(self):
        for kwargs in [dict(a=0),dict(b=np.nan),dict(h=-1),dict(ks=0)]:
            args=dict(a=1,b=1,h=.1,material=al_al2o3(1));args.update(kwargs)
            with self.assertRaises(ValueError): NavierFSDTPlate(**args)
        for pair in [(0,1),(1,1.5),(True,1)]:
            with self.assertRaises(ValueError): self.p.matrices(*pair)
        for loads in [{(1,1):np.nan},{(1,1):1j},{(0,1):1},{'x':1}]:
            with self.assertRaises(ValueError): self.p.bending_modal(loads)
        with self.assertRaises(ValueError): self.p.bending(0,'typo')
        with self.assertRaises(ValueError): self.p.bending(1,n_terms=0)
        with self.assertRaises(ValueError): self.p.bending_modal({},method='typo')

    def test_exact_moments_against_adaptive_quadrature(self):
        for g in [0,.01,.5,2,8,100]:
            mat=al_al2o3(g)
            for prop in ['E','rho']:
                for k in range(3):
                    h=.071
                    numerical=quad(lambda t:getattr(mat,prop)((t-.5)*h,h)*(t-.5)**k,0,1,
                                   epsabs=1e-7,epsrel=1e-11)[0]*h**(k+1)
                    scale=mat.moment(prop,0,h)*h**k
                    self.assertLess(abs(numerical-mat.moment(prop,k,h))/scale,2e-10)

    def test_closed_form_all_coefficients_and_equilibrium(self):
        rng=np.random.default_rng(37)
        for g,ratio,aspect in [(0,100,.5),(.02,5,2),(1,20,1),(8,100,2)]:
            p=NavierFSDTPlate(1,aspect,1/ratio,al_al2o3(g))
            loads={(m,n):rng.normal() for m in range(1,6) for n in range(1,6)}
            a=p.bending_modal(loads);b=p.bending_modal(loads,method='closed_form')
            for key in loads:
                assert_allclose(a.coeffs[key],b.coeffs[key],rtol=2e-10,atol=1e-25)
                K,_=p.matrices(*key);f=np.array([0,0,loads[key],0,0])
                residual=K@b.coeffs[key]-f
                self.assertLess(np.linalg.norm(residual)/max(1,abs(loads[key])),1e-9)

    def test_callable_projection_known_modes_and_uniform(self):
        p=self.p
        projected=p.project_load(lambda x,y:2*np.sin(2*np.pi*x/p.a)*np.sin(3*np.pi*y/p.b),5,32)
        for key,q in projected.items(): self.assertAlmostEqual(q,2 if key==(2,3) else 0,places=12)
        uniform=p.project_load(lambda x,y:1,5,32)
        for key,q in uniform.items(): self.assertAlmostEqual(q,p.load_coefficient(1,*key,'uniform'),places=12)
        with self.assertRaises(ValueError):
            p.project_load(lambda x,y:np.ones_like(x,dtype=complex)*(1+2j))
        with self.assertRaises(ValueError):
            p.project_load(lambda x,y:np.full_like(x,np.nan))

    def test_grid_points_and_derivatives(self):
        p=self.p;s=p.bending_modal({(1,2):.7,(3,1):-1})
        X,Y,F=s.field_grid(17,23);points=s.fields(X,Y)
        for name in F: assert_allclose(F[name],points[name],atol=1e-24,rtol=1e-12)
        x,y=.37,.51;eps=1e-6
        dx=s.fields(x,y,(1,0));dy=s.fields(x,y,(0,1))
        for name in F:
            fx=(s.fields(x+eps,y)[name]-s.fields(x-eps,y)[name])/(2*eps)
            fy=(s.fields(x,y+eps)[name]-s.fields(x,y-eps)[name])/(2*eps)
            assert_allclose(dx[name],fx,rtol=2e-7,atol=1e-20)
            assert_allclose(dy[name],fy,rtol=2e-7,atol=1e-20)

    def test_essential_and_natural_boundary_conditions(self):
        p=self.p;s=p.bending_modal({(1,2):.7,(3,1):-1,(2,2):.3});S=p.S
        for x in [0,p.a]:
            y=np.linspace(0,p.b,23);f=s.fields(x,y);dx=s.fields(x,y,(1,0));dy=s.fields(x,y,(0,1))
            for name in ['v0','w0','phiy']: assert_allclose(f[name],0,atol=1e-24)
            Nx=S.A11*dx['u0']+S.A12*dy['v0']+S.B11*dx['phix']+S.B12*dy['phiy']
            Mx=S.B11*dx['u0']+S.B12*dy['v0']+S.D11*dx['phix']+S.D12*dy['phiy']
            assert_allclose(Nx,0,atol=1e-12);assert_allclose(Mx,0,atol=1e-12)
        for y in [0,p.b]:
            x=np.linspace(0,p.a,23);f=s.fields(x,y);dx=s.fields(x,y,(1,0));dy=s.fields(x,y,(0,1))
            for name in ['u0','w0','phix']: assert_allclose(f[name],0,atol=1e-24)
            Ny=S.A12*dx['u0']+S.A11*dy['v0']+S.B12*dx['phix']+S.B11*dy['phiy']
            My=S.B12*dx['u0']+S.B11*dy['v0']+S.D12*dx['phix']+S.D11*dy['phiy']
            assert_allclose(Ny,0,atol=1e-12);assert_allclose(My,0,atol=1e-12)

    def test_load_superposition_and_zero_load(self):
        a=self.p.bending_modal({(1,2):2});b=self.p.bending_modal({(2,1):-3})
        c=self.p.bending_modal({(1,2):2,(2,1):-3})
        for name in c.fields(.3,.4):
            assert_allclose(c.fields(.3,.4)[name],a.fields(.3,.4)[name]+b.fields(.3,.4)[name])
        for value in self.p.bending_modal({}).fields(.3,.4).values(): self.assertEqual(value,0)

    def test_generalized_eigen_residual_and_mass_orthogonality(self):
        for pair in [(1,1),(2,3),(5,4)]:
            K,M=self.p.matrices(*pair);e=self.p.eigenpairs(*pair)
            assert_allclose(e.vectors.T@M@e.vectors,np.eye(5),atol=2e-13)
            for j in range(5):
                v=e.vectors[:,j];lam=e.omega[j]**2
                err=np.linalg.norm(K@v-lam*M@v)/(np.linalg.norm(K@v)+lam*np.linalg.norm(M@v))
                self.assertLess(err,1e-10)
            self.assertGreater(e.transverse_participation[e.flexural_index],.5)

    def test_global_flexural_scan_and_homogeneous_decoupling(self):
        for aspect in [.5,1,2]:
            p=NavierFSDTPlate(1,aspect,.05,al_al2o3(3))
            a=p.flexural_modes(4,5);b=p.flexural_modes(6,5)
            assert_allclose([x['omega'] for x in a],[x['omega'] for x in b],rtol=1e-12)
            self.assertAlmostEqual(a[0]['omega']/p.fundamental_frequency(),1,places=10)
        p=NavierFSDTPlate(1,1,.1,isotropic(70e9,2702))
        self.assertEqual(p.S.B11,0);self.assertEqual(p.S.I1,0)
        s=p.bending(1)
        self.assertEqual(s.coeffs[1,1][0],0);self.assertEqual(s.coeffs[1,1][1],0)


if __name__=='__main__': unittest.main(verbosity=2)
