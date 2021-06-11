// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
// mpirun -np 4 valgrind magnetodynamic_inductive -m torus2.msh --petscopts rc_ex10p_mfop

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>
#include <math.h>
#ifndef MFEM_USE_PETSC
#error This example requires that MFEM is built with MFEM_USE_PETSC=YES
#endif

using namespace std;
using namespace mfem;

class Hform_Operator;
class SuperconductorEJIntegrator; 
class SurfaceFluxLFIntegrator;
class PreconditionerFactory;

class ParDiscreteInterpolationOperator : public ParDiscreteLinearOperator
 {
  public:
    ParDiscreteInterpolationOperator(ParFiniteElementSpace *dfes,
                                    ParFiniteElementSpace *rfes)
   : ParDiscreteLinearOperator(dfes, rfes) {}
    virtual ~ParDiscreteInterpolationOperator();
 };
ParDiscreteInterpolationOperator::~ParDiscreteInterpolationOperator()
 {}

class ParDiscreteCurlOperator : public ParDiscreteInterpolationOperator
 {
   public:
     ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
                             ParFiniteElementSpace *rfes);
 };
ParDiscreteCurlOperator::ParDiscreteCurlOperator(ParFiniteElementSpace *dfes,
                                                 ParFiniteElementSpace *rfes)
  : ParDiscreteInterpolationOperator(dfes, rfes)
 {
 this->AddDomainInterpolator(new CurlInterpolator);
 }

class MagnetodynamicSolver : public TimeDependentOperator
{
protected:
   VisItDataCollection * visit_dc_;

   ParFiniteElementSpace  *HCurlFESpace, *HDivFESpace;
   ParBilinearForm  * Hform_LHS_linear, * Hform_massH;
   ParNonlinearForm * Hform_LHS_nonlinear;

   HypreParVector *H_t;  // Current value of the magnetic field DoFs
   ParGridFunction * H_t1,* H_t2;   // H field at time step 1, step 2
   Vector  rho_L, rho_NL, I_direction;
   VectorConstantCoefficient *Current_direction;
   PetscNonlinearSolver *petsc_solver;
   PreconditionerFactory *J_factory;
   Hform_Operator *Hform_oper;
   Coefficient       * muCoef_ ; //  permeability Coefficient
   VectorCoefficient * H_BCCoef_;   // Vector Potential BC Function
   PWConstCoefficient *rhoCoef_L, *rhoCoef_NL;

   int myid;
   double  (*muInv   ) (const Vector&);
   double  (*ItrFunc ) (double);
   void    (*H_BCFunc) (const Vector&, double, Vector&);
   double  (*muInv_   ) (const Vector&);
   double  (*ItrFunc_ ) (double);
   void    (*H_BC) (const Vector&, double, Vector&);
   Array<int> ess_bdr; // boundary marker for outer boundaries
   Array<int> ess_domain;  
   Array<int> *ess_tdof_list;
   Array<int> ess_bdr_cs; // boundary marker for conductor cross sections 
public:
   MagnetodynamicSolver(ParFiniteElementSpace & HCurlFESpace_, ParFiniteElementSpace & HDivFESpace_,
                        Array<int> &ess_bdr_, Array<int> &ess_domain_,
                         double  (*muInv   ) (const Vector&),
                         double  (*ItrFunc ) (double),
                         void    (*H_BCFunc) (const Vector&, double, Vector&) );
   
   HYPRE_Int GetProblemSize();
   void PrintSizes();
   Vector & GetHfield() { return *H_t; } // define the field H_ to be returned from ODE_solver->step() 
   void SetInitial_Hfield();
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);
   void SyncGridFuncs();
   void RegisterVisItFields(VisItDataCollection & visit_dc);
   void WriteVisItFields(int it = 0);  
   virtual ~MagnetodynamicSolver();
};

// parameter Function

double muInv(const Vector & x) {const double mu0_ = 1.2566e-6; return mu0_; }

// time-dependent boundary condition function on H

double ramp_current1(double t)
{  
   double  Itr = 10 * t ; // current is ramped up with time from zero 10A/s
   return  Itr;
}
double ItrFunc( double t) { ramp_current1(t); }
// Time-dependent boundary condition, dH/dt, derived from known time-dependent background field H(t) value
void
H_BCFunc(const Vector &x, double t, Vector &H_bc)
{
  const double mu0_ = 1.2566e-6;
  // H_bc = 0.0; // can be changed to time dependent, or, dH/
   H_bc.SetSize(3);
   H_bc(0) = 0;
   H_bc(1) = 0;
   H_bc(2) = t* 0.01/mu0_;
}


int main(int argc, char *argv[])
{
// 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

// Parse command-line options.
   const char *mesh_file = "../torus2.msh";
   int Order = 1;
   int serial_ref_levels = 1;
   int parallel_ref_levels = 0;
   bool visit = true;
   double dt = 1.0e-1;
   double dtsf = 0.95;
   double ti = 0.0;
   double ts = 0.5;
   double tf = 10.0;
   const char *petscrc_file = "";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&petscrc_file, "-petscopts", "--petscopts",
                  "PetscOptions file to use.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }
   // initilize PETSC
   MFEMInitializePetsc(NULL,NULL,petscrc_file,NULL);

   // Read the (serial) mesh from the given mesh file on all processors.  We can
   // handle triangular, quadrilateral, tetrahedral, hexahedral, surface and
   // volume meshes with the same code.

   Mesh *mesh = new Mesh(mesh_file, 1, 1);

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   FiniteElementCollection *fec_ND; 
   FiniteElementCollection *fec_RT; 
   fec_ND= new ND_FECollection(Order, pmesh->Dimension());
   fec_RT= new RT_FECollection(Order, pmesh->Dimension());
   ParFiniteElementSpace HCurlFESpace(pmesh,fec_ND);
   ParFiniteElementSpace HDivFESpace(pmesh,fec_RT);
   int current_num =1;
   Array<int> ess_bdr_(pmesh->bdr_attributes.Max()); //all boundary attributes in mesh
   Array<int> ess_domain_(pmesh->attributes.Max()); // all domain attributes in mesh

   int size_Hcurl =  HCurlFESpace.GetTrueVSize();
   cout << "Number of local H(Curl) unknowns: " << size_Hcurl << endl; 
   HYPRE_Int size_rt = HDivFESpace.GlobalTrueVSize();
   cout << "Number of global H(Div) unknowns: " << size_rt << endl;
   // Create the Electromagnetic solver
   MagnetodynamicSolver MagnetEM(HCurlFESpace, HDivFESpace, ess_bdr_, ess_domain_, muInv, ItrFunc, H_BCFunc);
   cout << "Start initialization Magnet solver." << endl;
   
   // Display the current number of DoFs in each finite element space
   MagnetEM.PrintSizes();
   // Set the initial conditions for both the current density J and magnetic fields H
   MagnetEM.SetInitial_Hfield();
   // Set the largest stable time step
   double dtmax = 0.001 ;

   // Create the ODE solver
   BackwardEulerSolver BESolver;
   BESolver.Init(MagnetEM);
   cout << "Initialization ODE solver finished." << endl;
   // Initialize VisIt visualization
   VisItDataCollection visit_dc("Magnetodynamic", pmesh);
   double t = ti;
   MagnetEM.SetTime(t);

   if ( visit )
   {
      MagnetEM.RegisterVisItFields(visit_dc);
   }
   // Write initial fields to disk for VisIt
   if ( visit )
   {
      MagnetEM.WriteVisItFields(0);
   }
   // The main time evolution loop.  
   int it = 1;
   t = t+dt*it;
   while (t < tf)
   {
      // Run the simulation until a snapshot is needed
       BESolver.Step(MagnetEM.GetHfield(), t, dt);  // Step() includes t += dt     H = H +dHdt*dt

      // Update local DoFs with current true DoFs
       MagnetEM.SyncGridFuncs();

      // Write fields to disk for VisIt
      if ( visit )
      {
         MagnetEM.WriteVisItFields(it);
      }
   it++;  
   }
   delete fec_ND;
   delete fec_RT;
   delete pmesh;
  MFEMFinalizePetsc();
  MPI_Finalize();
   return 0;
}

class Hform_Operator : public Operator
{
private:
   ParBilinearForm *LHS_linear, *LHS_massH;
   ParNonlinearForm *LHS_nonlinear;
   mutable HypreParMatrix *Jacobian;
   double dt; 
   const Vector *H_rhs, *currents;
   const Array<int> ess_tdof_list ;
   //DenseMatrix ;
   int myid;
public:
   Hform_Operator(ParNonlinearForm *LHS_nonlinear_,ParBilinearForm *LHS_linear_,ParBilinearForm *LHS_massH_,
   const Array<int> *ess_tdof_list_ );
   /// Set current dt, v, x values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *H_);
   /// Compute y = H(x + dt (v + dt k)) + M k + S (v + dt k).
   virtual void Mult(const Vector &k, Vector &Residual) const;
   /// Compute J = M + dt S + dt^2 grad_H(x + dt (v + dt k)).
   virtual Operator &GetGradient(const Vector &k) const;
   virtual ~Hform_Operator();
};
//
Hform_Operator::Hform_Operator(     
   ParNonlinearForm *LHS_nonlinear_,ParBilinearForm *LHS_linear_,ParBilinearForm *LHS_massH_,
   const Array<int> *ess_tdof_list_ )
   : Operator(LHS_linear_->ParFESpace()->TrueVSize()), LHS_nonlinear(LHS_nonlinear_), LHS_linear(LHS_linear_), 
   LHS_massH(LHS_massH_), dt(0.0), ess_tdof_list(*ess_tdof_list_)
{   
  cout << "Hform started " << endl;
  cout << "Hform constructed " << endl;
}

void Hform_Operator::SetParameters(double dt_, const Vector *H_)
{
   dt = dt_;
   H_rhs = H_; 
}

void Hform_Operator::Mult(const Vector &k, Vector &Residual) const // k should be the trueDofs
{ 
   cout << "Mult start"  <<endl;
   Vector y(k.Size()), w(k.Size());
   LHS_nonlinear->Mult(k, y); 
   
   LHS_linear->TrueAddMult(k, y, 1.0);
   add(k, -1,*H_rhs , w);
   LHS_massH->TrueAddMult(w, y, 1.0/dt);   
   cout << "nonlinear Mult updated" << endl;
  
   y.SetSubVector(ess_tdof_list, 0.0);  
   Residual = y; 
   cout << "residual norm " << Residual.Norml2() <<endl;
   cout << " Mult finished" << endl;
}

Operator &Hform_Operator::GetGradient(const Vector &k) const
{  
   cout << " Jacobain started " << endl;
   //delete Jacobian;  
   SparseMatrix *localJ = Add(1.0, LHS_linear->SpMat(), 1.0/dt, LHS_massH->SpMat()); // curlcurl(H)+H/dt

   localJ->Add(1.0, LHS_nonlinear->GetLocalGradient(k));
   // if we are using PETSc, the HypreParCSR Jacobian will be converted to
   // PETSc's AIJ on the fly  
   Jacobian = LHS_massH->ParallelAssemble(localJ);
   delete localJ; 
   HypreParMatrix *Je = Jacobian->EliminateRowsCols(ess_tdof_list);  
   //delete Je;
   cout << "Jacobian finished " <<endl;
   return *Jacobian;
}

Hform_Operator::~Hform_Operator()
{
delete Jacobian; 
}

/** Auxiliary class to provide preconditioners for matrix-free methods */
class PreconditionerFactory : public PetscPreconditionerFactory
{
private:
   // const ReducedSystemOperator& op; // unused for now (generates warning)

public:
   PreconditionerFactory(const Hform_Operator& op_, const string& name_)
      : PetscPreconditionerFactory(name_) /* , op(op_) */ {}
   virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle&);
   virtual ~PreconditionerFactory() {}
};
// This method gets called every time we need a preconditioner "oh"
// contains the PetscParMatrix that wraps the operator constructed in
// the GetGradient() method (see also PetscSolver::SetJacobianType()).
// In this example, we just return a customizable PetscPreconditioner
// using that matrix. .
Solver* PreconditionerFactory::NewPreconditioner(const mfem::OperatorHandle& oh)
{   
   PetscParMatrix *pP;
   oh.Get(pP);
   cout << " use PCfactory " << endl;
   return new PetscPreconditioner(*pP,"jfnk_");
}
// the Linearform Integrator to perform the normal flux on the an internal interface, used to impose current 
class SurfaceFluxLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   VectorCoefficient &Q;
   int oa, ob;
public:
   /// Constructs a boundary integrator with a given Coefficient QG
   SurfaceFluxLFIntegrator(VectorCoefficient &QG, int a = 1, int b = 1)
      : Q(QG), oa(a), ob(b) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el, ElementTransformation &Tr, Vector &elvect);
   using LinearFormIntegrator::AssembleRHSElementVect;
};
void SurfaceFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Qvec;
   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      Q.Eval(Qvec, Tr, ip);

      el.CalcPhysShape(Tr, shape);
      double area = nor.Norml2();
      double prod = Qvec*nor;
      double signedArea = area * ((fabs(prod) < 1e-4 * fabs(area)) ? 0.0 :
				  copysign(1.0, prod));     
      elvect.Add(ip.weight*signedArea, shape);
   }
}

class SuperconductorEJIntegrator : public NonlinearFormIntegrator
{
 protected:
   Coefficient *Q;
 private:  
   DenseMatrix curlshape, curlshape_dFt, Jac;
   Vector J,vec,pointflux;
   double E0=0.0001;
   int n=20;
   double Jc=100000000;  
 public: 
   SuperconductorEJIntegrator(Coefficient &m) 
   :Q(&m)   { }
   virtual void AssembleElementGrad(const FiniteElement &el,ElementTransformation &Ttr, const Vector &elfun, DenseMatrix &elmat);
   virtual void AssembleElementVector(const FiniteElement &el,ElementTransformation &Ttr, const Vector &elfun, Vector &elvect);  
};

void SuperconductorEJIntegrator::AssembleElementGrad(const FiniteElement &el,ElementTransformation &Ttr, 
                                                     const Vector &elfun, DenseMatrix &elmat)
   { 
    int nd = el.GetDof(),dim = el.GetDim(); 
    Vector J(dim);
    DenseMatrix curlshape(nd,dim), curlshape_dFt(nd,dim); // both trial and test function in Nedelec space, represented with curlshape
    elmat.SetSize(nd*dim);
    elmat = 0.0;
    double w;
    
    const IntegrationRule *ir = IntRule;
    if (!ir)
     {
       ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
     }
    for (int i = 0; i < ir->GetNPoints(); i++)
     {
       const IntegrationPoint &ip = ir->IntPoint(i);
       Ttr.SetIntPoint(&ip);
       w = ip.weight / Ttr.Weight();
       w *= Q->Eval(Ttr, ip);// multiply the PWconstantcoefficient
       el.CalcCurlShape(ip, curlshape); // curl operation on the shape function
       MultABt(curlshape, Ttr.Jacobian(), curlshape_dFt); // the curl operation of H(curl) sapce:  H(div)  u(x) = (J/w) * uh(xh)
       curlshape.MultTranspose(elfun, J); // compute the current density J
      
       double J_norm = J.Norml2();
       double J_de = E0/Jc*(n-1)*pow((J_norm/Jc), n-2); // derivative factor (n-1)*E0/Jc*(CurlH.Norm/Jc)^(n-2)
      // the transpose may be needed AtA rather than AAt
       AddMult_a_AAt(w*J_de, curlshape_dFt, elmat); // (Curl u, curl v)*J_de*w  
     }
   } 

void SuperconductorEJIntegrator::AssembleElementVector(const FiniteElement &el,ElementTransformation &Ttr, 
                                                       const Vector &elfun, Vector &elvect)
   {
    int nd = el.GetDof(), dim = el.GetDim(); 
    DenseMatrix curlshape(nd,dim); // both trial and test function in Nedelec space, represented with curlshape
    double w;
    J.SetSize(dim); 
    Jac.SetSize(dim); 
    pointflux.SetSize(dim);
    vec.SetSize(dim);
    elvect.SetSize(nd);
    elvect = 0.0;    
    //cout << "elfun size " <<  elfun.Size() << endl; 
    //cout << "Densemtrix row col " << nd <<" Elfun size " << elfun.Size() << endl;
    const IntegrationRule *ir = IntRule;
    if (!ir)
     {
       ir = &(IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + 3)); // <---
     } 
    for (int i = 0; i < ir->GetNPoints(); i++)
     {
       const IntegrationPoint &ip = ir->IntPoint(i);
       Ttr.SetIntPoint(&ip);
       w = ip.weight / Ttr.Weight();
       w *= Q->Eval(Ttr, ip);     // multiply the PWconstantcoefficient  
       el.CalcCurlShape(ip, curlshape); // curl operation on the shape function
     
       curlshape.MultTranspose(elfun, J); // compute the current density J
       Jac = Ttr.Jacobian(); // mapping Jacobian to the reference element
           
       curlshape.MultTranspose(elfun, vec); //
       Jac.MultTranspose(vec, pointflux);
       //double J_norm=  pow(J[0],2) + pow(J[1],2) ;
       double J_norm = J.Norml2();
      
       double J_f = E0/Jc*pow((J_norm/Jc), n-1); //  factor E0/Jc*(CurlH.Norm/Jc)^(n-1)
       //cout << "current level " <<  J_f << endl; 
       pointflux *= w*J_f;
       Jac.Mult(pointflux, vec);
       curlshape.AddMult(vec, elvect); // (Curl u, curl v)*J_f*w 
     }
     
   }

MagnetodynamicSolver::MagnetodynamicSolver (ParFiniteElementSpace & HCurlFESpace_, ParFiniteElementSpace & HDivFESpace_,
                        Array<int> &ess_bdr_, Array<int> &ess_domain_, 
                         double  (*muInv   ) (const Vector&),
                         double  (*ItrFunc ) (double),
                         void    (*H_BCFunc) (const Vector&, double, Vector&)
                        )
   : HCurlFESpace(&HCurlFESpace_), HDivFESpace(&HDivFESpace_),ess_bdr(ess_bdr_),ess_domain(ess_domain_),
   visit_dc_(NULL),muInv_(muInv),H_BC(H_BCFunc),ItrFunc_(ItrFunc),H_t(NULL),
   Hform_LHS_linear(NULL),Hform_LHS_nonlinear(NULL), Hform_massH(NULL),ess_tdof_list(NULL),Hform_oper(NULL), J_factory(NULL)
           
{    
   MPI_Comm_rank(HCurlFESpace->GetComm(), &myid);
   cout << "Creating Magnet Solver" << endl;
   const double rel_tol = 1e-8;   
   // Select surface attributes for Dirichlet BCs
   Array<int> ess_tdof_list_, ess_bdr_cs(ess_bdr.Size());
   int brs = ess_bdr.Size();
   int dms = ess_domain.Size();
   cout << "Number of domains " << dms << endl;
   cout << "Number of boundaries " << brs << endl;
   ess_bdr = 1.0;
   ess_bdr[ess_bdr.Size()-1] = 0;// the last surface attribute is the internal interface (cross section) 
   // ess_bdr_ makes the outer boundary, ess_bdr_cs marks the inner interface (cross section)
   ess_bdr_cs = 0; 
   ess_bdr_cs[ess_bdr.Size()-1] = 1;
   // Setup various coefficients
   HCurlFESpace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list_);
   ess_tdof_list = &ess_tdof_list_;
   cout << "essential true dofs num " << ess_tdof_list->Size() << endl;
   // magnetic field H on the outer surface   define a coefficient from function 
   // define the time-dependent coefficient for the BC, with the BC function
   Vector rho_L(ess_domain.Size()); // resistivity for the linear domains, superconducting domains set to 0
   rho_L = 15.0;
   rho_L(0) = 1*100; 
   rhoCoef_L = new PWConstCoefficient(rho_L);
   Vector rho_NL(ess_domain.Size()); // resistivity for nonlinear domains, non-superconducting domains set to 0
   rho_NL=1;
   rho_NL(0) =0.0;
   rhoCoef_NL = new PWConstCoefficient(rho_NL);

   muCoef_ = new FunctionCoefficient(muInv_);   // coefficient for permeability
   H_BCCoef_ = new VectorFunctionCoefficient(3,H_BC);  // coefficient for time-dependent boundary condition for H 

   cout << "Creating Coefficients" << endl;  
   // Bilinear Forms
   Hform_LHS_linear = new ParBilinearForm(HCurlFESpace); // H_formulation left hand side, the linear component
   Hform_LHS_linear->AddDomainIntegrator(new CurlCurlIntegrator(*rhoCoef_L)); // the curl_curl operation on LHS of the linear part curl(curl H2)
   Hform_LHS_linear->Assemble();
   Hform_LHS_linear->Finalize();
   cout << "Creating bilinear objects" << endl;
   
   Hform_massH = new ParBilinearForm(HCurlFESpace);////The mass operation, compute the H2/dt
   Hform_massH->AddDomainIntegrator( new VectorFEMassIntegrator(*muCoef_));   
   Hform_massH->Assemble();
   Hform_massH->Finalize();

   Hform_LHS_nonlinear = new ParNonlinearForm (HCurlFESpace); // H_formulation left hand side, the nonlinear component
   Hform_LHS_nonlinear->AddDomainIntegrator(new SuperconductorEJIntegrator(*rhoCoef_NL)); // the curl_curl operation with nonlinear E-J
   Hform_LHS_nonlinear->SetEssentialTrueDofs(*ess_tdof_list);   
  
   cout << "Hcurl true size every process" << HCurlFESpace->GetTrueVSize() << endl;
   H_t1 = new ParGridFunction(HCurlFESpace);// the solution vector for H field
   H_t2 = new ParGridFunction(HCurlFESpace);// the solution vector for H field   
 //.......................................................................................
   // for total net transport currents in each conductor  
 // use PETS nonlinear solver 
   Hform_oper = new Hform_Operator (Hform_LHS_nonlinear,Hform_LHS_linear, Hform_massH, ess_tdof_list);
   cout << "H form Operator created" << endl;
   cout << "comm is " << HCurlFESpace->GetComm() << endl;
   petsc_solver = new PetscNonlinearSolver(HCurlFESpace->GetComm(), *Hform_oper);
   // construct the preconditioner for JFNK solution
   J_factory = NULL;
   J_factory = new PreconditionerFactory(*Hform_oper, "JFNK preconditioner");

   petsc_solver->SetPreconditionerFactory(J_factory);
   //petsc_solver->SetJacobianType(Operator::PETSC_MATHYPRE );
   petsc_solver->SetPrintLevel(1); // print Newton iterations
   petsc_solver->SetRelTol(rel_tol);
   petsc_solver->SetAbsTol(0.0);
   petsc_solver->SetMaxIter(10);
   //............................................................................................   
}

MagnetodynamicSolver::~MagnetodynamicSolver()
{ 
   delete J_factory;
   delete petsc_solver;
   delete Hform_oper;
   delete H_t1;  
   delete H_t2; 
   //delete current_constraint; 
   delete muCoef_;
   delete H_BCCoef_;
   delete rhoCoef_NL;
   delete rhoCoef_L;
 
   delete Hform_LHS_linear;
   delete Hform_LHS_nonlinear;
   delete Hform_massH; 
   delete H_t;
   delete HCurlFESpace;
   delete HDivFESpace;
}

HYPRE_Int
MagnetodynamicSolver::GetProblemSize()
{
   return HCurlFESpace->GlobalTrueVSize();
}

void
MagnetodynamicSolver::PrintSizes()
{
   HYPRE_Int size_nd = HCurlFESpace->GlobalTrueVSize();
   HYPRE_Int size_rt = HDivFESpace->GlobalTrueVSize();
      cout << "Number of H(Curl) unknowns: " << size_nd << endl;
      cout << "Number of H(Div)  unknowns: " << size_rt << endl;
}

void
MagnetodynamicSolver::SetInitial_Hfield()
{
   *H_t1 =0.0;
   H_t =H_t1->ParallelProject(); // initialize *H_t=0 
   cout << "H field initialized " << endl;
}

void  MagnetodynamicSolver::ImplicitSolve(double dt, const Vector &H, Vector &dHdt)  // perform the calculation of H and J in every time step
{
   cout << "Time t= " << t << endl;
   cout << "H norm " << H.Norml2() << endl;// H is the H field values from last time step
   H_BCCoef_->SetTime(t); // update the BC of H to current time step, 't' is member data from mfem::TimeDependentOperator
   *H_t2 = 0.0;                    // initialize the solution vector for the current time step to 0
   H_t2->ProjectBdrCoefficient(*H_BCCoef_, ess_bdr);   //    add Dirichlet BC into solution vector 
   Vector H_rhs(H.Size()) ; // empty vector 
   H_rhs = H;
   cout << "H_rhs norm" << H_rhs.Norml2() << endl;
   HypreParVector  *H_par= H_t2->GetTrueDofs();// initilized the solution vector with BC 
   cout << "BC values initialized " << endl;  
    
  // Vector h_lhs= *H_par->GlobalVector();; // the x vector LHS

   Hform_oper->SetParameters(dt, &H_rhs);  
   Vector zero;
   petsc_solver->Mult(zero,*H_par); // Mult(b,x);
      MFEM_VERIFY(petsc_solver->GetConverged(), "Newton solver did not converge.");
   
   add(*H_par,-1.0, H, dHdt) ;
}
void
MagnetodynamicSolver::SyncGridFuncs()
{
   H_t1->Distribute(*H_t); // H_ returned by ODE_solver->step()
}

void
MagnetodynamicSolver::RegisterVisItFields(VisItDataCollection & visit_dc)
{
   visit_dc_ = &visit_dc;

   visit_dc.RegisterField("H", H_t1);
   //visit_dc.RegisterField("J", J_t1); add the J distribution later
   // visit_dc.RegisterField("B", B_);
}

void
MagnetodynamicSolver::WriteVisItFields(int it)
{
   if ( visit_dc_ )
   {
      cout << "Writing VisIt files ..." << flush; 

      HYPRE_Int prob_size = this->GetProblemSize();
      visit_dc_->SetCycle(it);
      visit_dc_->SetTime(prob_size);
      visit_dc_->Save();

       cout << " done." << endl; 
   }
}