# KBAI Lab 1 Concept Guide

## Purpose

This guide summarizes the ebook concepts needed for every activity in
`Lab 1.tex`. Use it to prepare prompts, evaluate Jill Watson's responses, and
write the report in your own words.

Source: [Knowledge-Based Artificial Intelligence: A Cognitive Approach to Artificial Intelligence](./kbai_ebook.pdf)
by Ashok Goel, David Joyner, and Steven Bryant.

## Page Number Note

The page numbers below use the printed ebook page numbers. The PDF viewer page
is consistently 22 pages later. For example, ebook page 34 is PDF page 56.

## Assignment Concept Map

| Lab section | Ebook chapter | Main concepts |
| --- | --- | --- |
| Activity A | Chapter 1 | Grand Challenges, Four Schools of AI, Explanation Enigma |
| Activity B | Chapters 3 and 4 | Generate and Test, Smart Generator, Smart Tester, Means-Ends Analysis, local optima |
| Activity C | Chapter 6 | Frames, slots, fillers, default values, exceptions, Top-Down Processing |
| Comparison analysis | Chapter 2 | Semantic Networks, Lexicon, Structural Specification, Semantics |
| Socratic audit | Chapters 1-4 and 6 | Review of the concepts above |

Use a fresh Jill Watson session for each activity, as required by the lab.

## Activity A: Four Schools and Grand Challenges

### What We Are Trying to Understand

We are testing whether Jill Watson can apply Chapter 1's course-specific
framework to a real AI system. It should classify the system thoughtfully and
connect its explanation to the Grand Challenges instead of giving a generic
description of AI.

### The Five Grand Challenges

Chapter 1 calls these the core conundrums or Grand Challenges of AI.

| Challenge | Clear explanation | Ebook page |
| --- | --- | --- |
| Resource Riddle | Agents have limited time, memory, and computing power, but many problems are computationally huge. | 2-3 (PDF 24-25) |
| Local Actions, Global Problems | Computation happens through local steps, but a valid or optimal solution may depend on global constraints. | 2-3 (PDF 24-25) |
| Logic Loophole | Computers are grounded in deduction, while real-world reasoning often requires induction or abduction. | 2-3 (PDF 24-25) |
| Dynamic World Dilemma | The world changes continually, while an agent's knowledge is finite and may become outdated. | 2-4 (PDF 24-26) |
| Explanation Enigma | Finding an answer is difficult; explaining and justifying the reasoning can be even more difficult. | 2-3 (PDF 24-25) |

The lab specifically asks about the **Explanation Enigma**. A strong Jill
Watson response should discuss whether the chosen AI application can make its
reasoning understandable and justifiable, not merely whether its prediction is
accurate.

### The Four Schools of AI

Russell and Norvig's framework uses two dimensions:

1. Is the focus on **thinking** or **acting**?
2. Is the goal to behave **rationally/optimally** or **like a human**?

This produces four schools:

| School | Meaning | Ebook example | Ebook page |
| --- | --- | --- | --- |
| Thinking Rationally | Reason toward an optimal result without needing to imitate human thought. | Many machine-learning algorithms | 10-11 (PDF 32-33) |
| Acting Rationally | Perform actions that achieve the best outcome. | Airplane autopilot | 11 (PDF 33) |
| Acting Human-like | Behave in a manner similar to a human. | Improvisational dancing robot | 11 (PDF 33) |
| Thinking Human-like | Use reasoning processes resembling human cognition. KBAI often leans toward this school. | Semantic Web technologies | 11 (PDF 33) |

The categories are not always rigid. The ebook notes that systems may span
multiple quadrants depending on which capability is being analyzed. For
example, an autonomous vehicle is better viewed primarily as acting rationally
when safe arrival matters more than imitating a human driver.

Relevant pages: 10-14 (PDF 32-36).

### Good Evaluation Questions

- Did Jill Watson select a sensible primary quadrant for the application?
- Did it explain why using both dimensions?
- Did it recognize any overlap between quadrants?
- Did it connect the application to the Explanation Enigma using Chapter 1's
  meaning: explanation and justification add difficulty beyond solving?

## Comparison Analysis: Semantic Networks

### What We Are Trying to Understand

We are comparing Jill Watson's ebook-grounded answer with a general LLM's
answer. The goal is to determine whether grounding produces a more rigorous,
course-specific representation.

### Core Idea

A Semantic Network is a knowledge representation that uses objects or concepts
and the relationships between them. It supports the KBAI
**represent-and-reason** approach: organize knowledge so that an agent can make
inferences.

Relevant pages: 27-28 (PDF 49-50).

### Three Components

| Component | Clear explanation | Ebook page |
| --- | --- | --- |
| Lexicon | The vocabulary of the representation. In a Semantic Network, these are nodes representing objects or concepts. | 34 (PDF 56) |
| Structural Specification | The permitted organization of the Lexicon. In a Semantic Network, these are directed links between nodes. | 34 (PDF 56) |
| Semantics | The meanings assigned to the links using application-specific labels. These meanings enable inference. | 34 (PDF 56) |

For the lab's cube-bricks-sphere scene, an illustrative Semantic Network should:

- Create separate nodes for the cube, both bricks, and the sphere.
- Separate nodes from relationships.
- Represent support relationships as directed, labeled links.
- Explain the direction of each relationship, such as `brick_1 -> supports -> cube`.

A good representation makes relationships explicit, exposes natural
constraints, avoids irrelevant detail, and remains transparent, concise,
complete, fast, and computable.

Relevant pages: 34-35 (PDF 56-57).

### Comparison Matrix Lens

Evaluate both answers using:

1. **Groundedness:** Does the answer use Lexicon, Structural Specification, and
   directed labeled links?
2. **Representational rigor:** Does it separate nodes from relationships?
3. **Reasoning transparency:** Does it explain what each link means and why its
   direction matters?

## Activity B: Problem-Solving Under Constraint

### What We Are Trying to Understand

This activity tests two distinct mechanisms:

1. Why Means-Ends Analysis can become trapped at a local optimum.
2. How Generate and Test can reduce wasted work through a Smart Generator.

Do not blend these mechanisms together in the report.

### Generate and Test

Generate and Test is a two-step problem-solving method:

1. Generate possible solutions or successor states.
2. Test them against the goal and problem constraints.

It is useful when knowledge is incomplete, computing resources are finite, and
reasoning methods are not infallible.

Relevant pages: 55-57 (PDF 77-79).

### Generator and Tester Responsibilities

| Component | Responsibility | Ebook page |
| --- | --- | --- |
| Simple Generator | Produces possible successor states indiscriminately. | 56 (PDF 78) |
| Simple Tester | Rejects generated states that violate obvious constraints. | 57 (PDF 79) |
| Smart Tester | Also rejects repeated, redundant, or otherwise unproductive states. | 63-64 (PDF 85-86) |
| Smart Generator | Avoids generating redundant or illegal states in the first place, reducing the Tester's workload. | 64-65 (PDF 86-87) |

The responsibility can sometimes be assigned to either component. A Tester may
reject a dead-end state after generation, or a sufficiently knowledgeable
Generator may avoid producing it. The practical design question is where the
pruning knowledge should live.

Relevant page: 66 (PDF 88).

### Means-Ends Analysis

Means-Ends Analysis (MEA) compares the current state with the goal state,
identifies differences, and greedily selects an operator that reduces those
differences.

- **Means:** the operator or action.
- **Ends:** reducing the difference between the current and goal states.
- **State space:** all configurations reachable by repeatedly applying valid
  operators.

Relevant pages: 73, 76, 80-82 (PDF 95, 98, 102-104).

### Why MEA Gets Stuck

MEA is greedy. A solution may require temporarily increasing a difference or
undoing apparent progress. A strict MEA strategy resists that move and can get
stuck at a local optimum or enter a loop.

In the ebook's four-block example, MEA makes moves that reduce the difference
count. It eventually reaches a state where every available next move appears
to move farther from the goal. Reaching the actual solution requires
reconsideration, backtracking, or a different strategy.

Relevant pages: 82-87 (PDF 104-109).

### Good Evaluation Questions

- Did Jill Watson correctly diagnose the local optimum?
- Did it explain why MEA's own greedy logic causes the trap?
- Did it keep MEA separate from Generate and Test?
- Did its Smart Generator prune illegal or unproductive moves before the
  Tester receives them?
- Did it explain the trade-off in assigning knowledge to the Generator versus
  the Tester?

## Activity C: Frame-Based Understanding

### What We Are Trying to Understand

We are testing whether Jill Watson can distinguish facts explicitly stated in
an ambiguous scenario from assumptions introduced by a Frame. The activity
also asks whether Jill Watson can explain how Top-Down Processing shaped those
assumptions.

### Frames, Slots, and Fillers

A Frame is a structured packet of knowledge. It organizes a familiar concept,
object, event, or action using:

- **Slots:** expected categories of information.
- **Fillers:** values placed into those slots.

For an `eating` Frame, slots may include subject, consumed object, location,
time, utensils, and the resulting state of the object. Some fillers come
directly from the sentence; other fillers are inferred using defaults.

Relevant pages: 127-130 (PDF 149-152).

### Three Main Characteristics

| Characteristic | Clear explanation | Ebook page |
| --- | --- | --- |
| Stereotypes | Frames encode generalized expectations about familiar concepts or situations. | 131 (PDF 153) |
| Default Values and Exceptions | A Frame supplies typical fillers when details are missing. Explicit facts or special cases can override those defaults. | 132 (PDF 154) |
| Inheritance Hierarchy | A general Frame passes common attributes to specialized Frames, which can add details or override defaults. | 133-134 (PDF 155-156) |

### Bottom-Up and Top-Down Processing

- **Bottom-Up Processing:** Start with facts stated in the scenario and fill
  slots from the available data.
- **Top-Down Processing:** Retrieve a familiar Frame from memory and use its
  expected slots and default values to interpret missing or ambiguous details.

Frames are efficient because they generate expectations quickly. They are also
limited: defaults can introduce assumptions or bias, and a Frame alone may not
resolve subtle ambiguity.

Relevant pages: 141-144 (PDF 163-166).

### Good Evaluation Questions

- Did Jill Watson label which fillers came directly from the scenario?
- Did it label which fillers were assumed defaults?
- Did it explain which Frame was retrieved and why?
- Did it describe how Top-Down Processing guided the assumptions?
- Did it acknowledge possible alternative interpretations?

## Socratic Audit Review Map

| Question | Concept to know | Reasoning | Relevant ebook page |
| --- | --- | --- | --- |
| 1 | Four Schools | An airplane autopilot is primarily an **Acting Rationally** agent because it acts in the world to achieve an optimal outcome. | 11 (PDF 33) |
| 2 | Semantic Networks | A Structural Specification defines the **directed, labeled links** between nodes. | 34 (PDF 56) |
| 3 | Generate and Test | A Smart Generator **prunes illegal or unproductive moves before** the Tester evaluates them. | 64-66 (PDF 86-88) |
| 4 | MEA local optimum | MEA can fail because a solution may require temporarily increasing a difference, which greedy difference reduction treats as counterproductive. | 82-87 (PDF 104-109) |
| 5 | Frames | A specific instance, such as a penguin that swims, **overrides** an inherited default value in its slot. | 132-133 (PDF 154-155) |

## Recommended Working Order

1. Choose an AI application and complete Activity A.
2. Design a Blocks World scenario and complete Activity B.
3. Design an ambiguous scenario and complete Activity C.
4. Run the Semantic Network comparison with Jill Watson and a general LLM.
5. Run the five-question Socratic audit.
6. Paste all six complete transcripts into `Lab 1.tex`.
7. Write the report sections using your observations from the transcripts.

## Transcript Submission Note

Reference discussion:
<https://edstem.org/us/courses/98722/discussion/8088626?answer=18711024>

Useful takeaway:

- Do not rely on the `verbatim` block if it causes long transcript lines to run
  past the page margin.
- The instructor clarified that you can drop the `verbatim` block and paste the
  transcript as plaintext.
- If Jill Watson provides images, include them separately only if they are
  needed to understand the conversation.
- Exporting the chat as PDF or Markdown may help, but availability of export
  options may vary.
