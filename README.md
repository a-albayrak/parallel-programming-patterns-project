# CENG444 - Fall 2024  
**FINAL PROJECT**  

---

The purpose of the project is to design, implement, and evaluate a **parallel programming solution** for a computational problem.  
You are required to apply **parallel programming paradigms**, demonstrate an understanding of **parallel design patterns**, and analyze performance improvements.  
You need to use **OpenMP parallel programming model** to implement a parallel solution for a computationally-heavy problem.

---

## Some highlights about the project:

- Your code should be functionally correct and potentially work better than the serial version.  
- You need to demonstrate which specific **parallel programming techniques and patterns** are used in your implementation.  
- You need to include specific **optimization methods** for those patterns discussed in class.  
  - For example:  
    - If you have a data-parallel part represented by the **Map** pattern, just using `#pragma omp parallel for` is not enough.  
    - You need to consider further optimizations and include them where appropriate.  
    - If no optimization is possible, you must justify it.

- You should include a **rigorous experimental analysis** in your report:  
  - Show how your code behaves  
  - Identify main **performance bottlenecks**  
  - Fix them through modifications/optimizations if they exist

- You need to perform a **scalability analysis**:
  - Analyze both **problem size** and **core counts**
  - Include profiling results and provide comments

- It is not mandatory to use **UHEM** (but highly recommended) as the experimental platform.
  - Your scalability analysis must include executions using **different numbers of cores/threads/schedule options**

- You will prepare a **report in IEEE conference paper format (at most 6-pages)**  
  [IEEE Template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn)

  - If needed, use an **Appendix** for:
    - Additional experimental results
    - Optional implementation details

---

## The report should include the following sections:

- **Problem Definition**: Your problem/algorithm  
- **Methodology**: Parallelization techniques and tools used  
- **Experimental Work**:
  - Experimental setup:
    - Hardware and software environment/specs
    - Input dataset (if any)
  - Reporting performance metrics (execution time, thread activity, memory usage)
  - Identification and explanation of **performance bottlenecks**
  - **Scalability analysis**
  - Discussion/comments about the results  
- **Future Work**: How the work could be extended

---

## [Link to IEEE template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn)

---
