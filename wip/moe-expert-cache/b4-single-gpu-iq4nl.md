# B4 single-GPU: Qwen3.8-Flash-Next IQ4_NL on one R9700 (gfx1201)

Date: 2026-09-30 (session 16).  Build: campaign `wip-moe-devmap-v2` tip `c5bbb7ee2` (r25 + cache).

Config: `HIP_VISIBLE_DEVICES=0`, `-m Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf`,
`-ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode auto --load-mode none -t 8`, cache
`MOE_EXPERT_CACHE_MIB=24576 MOE_EXPERT_CACHE_DEVMAP=1`.

Cache: 144 tables, **194/512 slots/table (37.9 % residency)**, arena 24.0 GiB; sample h ~0.84-0.85.

| test | result |
|---|---|
| 8K prefill () | **383.7 t/s** |
| essay-prompt prefill (~250 tok) | 15.3 t/s |
| 3000-token essay generation (warmed) | **34.6 t/s** (~87 s for 3000 tok) |
| short-run warm (8 tok, h~0.85) | 25.2 t/s at MIB=24576 |

The essay output (2245 words, 3000 tokens) follows.

```
# The Story of Computing: From Babbage to Machine Intelligence

### Introduction

In the grand tapestry of human history, few threads have woven themselves as tightly into the fabric of daily existence as the thread of computation. Today, we live in an age where the device in our pocket possesses more processing power than the supercomputers that guided Apollo 11 to the moon. We carry libraries of knowledge, instant communication channels, and complex simulation engines in our palms, often without a second thought to the mechanical, electrical, and logical revolutions that made such ubiquity possible. However, the story of computing is not merely a chronicle of faster chips and smaller transistors; it is a narrative of human ingenuity attempting to extend the reach of the human mind. It is a journey from the clatter of brass gears in a Victorian drawing room to the silent, humming logic gates of neural networks that can write poetry, diagnose disease, and play games of strategy at a superhuman level.

This story begins not with electricity, but with arithmetic. It spans centuries of theoretical abstraction, wartime necessity, corporate ambition, and scientific breakthrough. To understand where we are going with machine intelligence, we must first understand where we have been. We must trace the lineage of the computer from a theoretical engine of calculation to a universal machine of logic, then to a personal tool of creativity, and finally to an autonomous agent of thought. This essay will explore this evolution through distinct eras, examining the key figures, the pivotal technologies, and the shifting paradigms that have defined the field. It is a story of how we built machines to think, and in doing so, learned more about what it means to think ourselves.

### 1. The Mechanical Dream: Babbage, Lovelace, and the First Algorithms

Long before the first electron flowed through a circuit, the concept of the computer existed in the mind of a man who dreamed in brass and iron. Charles Babbage, an English mathematician and philosopher, is widely recognized as the "father of the computer." In the early 19th century, Babbage was frustrated by the frequency of errors in mathematical tables used for navigation and engineering. He believed these errors were due to human fallibility and proposed a solution: a machine that could perform calculations automatically, without fatigue or bias.

His first invention, the Difference Engine, was designed specifically to calculate polynomial functions using the method of finite differences. It was a marvel of mechanical engineering, consisting of thousands of interlocking gears, levers, and cams. While Babbage successfully built a small portion of the engine, demonstrating its functionality, the full-scale machine was never completed during his lifetime due to funding issues and the limitations of precision engineering of the era. However, the true significance of Babbage’s work lay not in the Difference Engine, but in his subsequent conceptual design: the Analytical Engine.

The Analytical Engine was a radical departure from previous calculating devices. It was not merely a calculator; it was a general-purpose computer. Babbage envisioned a machine that could accept input from punched cards (an idea borrowed from the Jacquard loom, which used cards to control patterns in weaving), store data in a "mill" (the central processing unit) and a "store" (memory), and execute a sequence of instructions. This architecture—the separation of input, storage, processing, and output—is the foundational structure of every modern computer.

Yet, it was Ada Lovelace, the daughter of the poet Lord Byron and a mathematician in her own right, who recognized the profound implications of Babbage’s design. In 1843, while translating an article on the Analytical Engine by Italian engineer Luigi Menabrea, Lovelace appended a set of notes that were longer and more insightful than the original article itself. In these notes, she described an algorithm intended to be processed by the Analytical Engine to calculate Bernoulli numbers. For this, she is often cited as the first computer programmer.

Lovelace’s genius lay in her ability to see beyond the numbers. She understood that because the machine could manipulate symbols, it could do more than just calculate. It could compose music, generate graphics, or manipulate language, provided that these things could be expressed in a logical form. She famously wrote that the engine "has no pretensions whatever to originate anything. It can do whatever we know how to order it to perform." This distinction—between a tool that executes commands and a mind that originates ideas—would echo through the next century and a half of computer science, setting the stage for the debate on artificial intelligence.

### 2. The Theoretical Foundation: Turing, Shannon, and the Birth of Logic

While Babbage and Lovelace built the physical dream of computing, it was the early 20th century that provided the theoretical bedrock upon which modern computers would stand. The transition from mechanical gears to abstract logic was driven by mathematicians who sought to define the limits of knowledge and calculation.

Alan Turing, a British mathematician and logician, stands as the central figure in this theoretical revolution. In his seminal 1936 paper, "On Computable Numbers, with an Application to the Entscheidungsproblem," Turing introduced the concept of the "Turing Machine." This was not a physical device but a mathematical model of computation. A Turing Machine consists of an infinite tape, a head that can read and write symbols, and a set of rules for moving and changing states. Turing proved that this simple model could simulate any algorithmic process, no matter how complex, provided it had enough time and memory. This established the concept of the "Universal Turing Machine," a machine capable of computing any computable function. This was the theoretical birth of the stored-program computer.

Turing’s work was not merely abstract; it was born of urgency. During World War II, Turing worked at Bletchley Park, the British codebreaking center. There, he contributed significantly to the cracking of the Enigma cipher used by the Nazi military. This work necessitated the development of electronic computing devices, such as the Colossus, which was used to break the Lorenz cipher. These machines were not general-purpose computers in the modern sense, but they demonstrated the power of electronic switching to perform high-speed logical operations.

Simultaneously, across the Atlantic, Claude Shannon, an engineer at Bell Labs, published his master’s thesis, "A Symbolic Analysis of Relay and Switching Circuits," in 1938. Shannon demonstrated that Boolean algebra—a branch of mathematics dealing with true/false values—could be applied to electrical switching circuits. This insight bridged the gap between abstract logic and physical engineering. It meant that logical operations (AND, OR, NOT) could be performed by electrical relays and, later, transistors. Shannon’s work provided the language for digital circuit design, allowing engineers to build machines that could process information logically rather than just numerically.

Together, Turing and Shannon laid the dual foundations of computer science: the theoretical limits of computation and the physical implementation of logical operations. With these foundations in place, the stage was set for the construction of the first electronic general-purpose computers.

### 3. The Vacuum Tube Era: Giants in the Room

The first generation of electronic computers emerged in the 1940s, born from the exigencies of war and the ambition of scientists. These machines were massive, occupying entire rooms, consuming vast amounts of electricity, and generating significant heat. They used vacuum tubes as their primary switching elements.

Among the most famous of these early machines was ENIAC (Electronic Numerical Integrator and Computer), completed in 1945 at the University of Pennsylvania. Designed by John Mauchly and J. Presper Eckert, ENIAC was built to calculate artillery firing tables for the U.S. Army. It contained 17,468 vacuum tubes, 70,000 resistors, and 10,000 capacitors. It weighed 30 tons and consumed 150 kilowatts of power. When ENIAC was turned on, lights in Philadelphia reportedly dimmed. Despite its bulk, ENIAC was revolutionary. It could perform 5,000 additions per second, a speed unimaginable for human calculators or mechanical devices.

However, ENIAC had a significant limitation: it was not a stored-program computer. To change its program, operators had to physically rewire the machine by plugging cables into panels and setting switches. This process was time-consuming and error-prone.

The concept of the stored-program computer was articulated most famously in the "First Draft of a Report on the EDVAC," written by John von Neumann in 1945. Von Neumann, a Hungarian-American mathematician, outlined an architecture where both the data and the instructions (the program) would be stored in the same memory unit. This allowed the computer to modify its own instructions and execute programs dynamically without physical rewiring. This "Von Neumann architecture," with its central processing unit (CPU), memory, and input/output devices, remains the fundamental design of most computers today.

The first stored-program computer was the Manchester Baby, built at the University of Manchester in 1948. It was a small, experimental machine, but it proved the concept. Following this, the EDSAC (Electronic Delay Storage Automatic Calculator) at Cambridge University became the first practical stored-program computer to run a program in 1949. These machines marked the transition from special-purpose calculators to general-purpose computers, capable of solving a wide variety of problems by simply loading different programs.

### 4. The Transistor Revolution: Smaller, Faster, Cheaper

The vacuum tube era was short-lived. Vacuum tubes were fragile, unreliable, and generated excessive heat, leading to frequent breakdowns. The solution came in 1947, when William Shockley, John Bardeen, and Walter Brattain at Bell Labs invented the transistor. The transistor was a semiconductor device that could act as a switch or an amplifier, performing the same functions as a vacuum tube but in a fraction of the size, with greater reliability, and at a much lower cost.

The invention of the transistor marked the beginning of the second generation of computers in the late 1950s. The IBM 7090, introduced in 1959, was one of the first commercially successful transistorized computers. These machines were smaller, faster, and more reliable than their vacuum-tube predecessors. They were primarily used by governments, universities, and large corporations for scientific calculations, data processing, and business applications.

The transistor also enabled the development of higher-level programming languages. In the 1950s, Grace Hopper, a rear admiral in the U.S. Navy and a computer pioneer, developed COBOL (Common Business-Oriented Language). Hopper had previously developed the first compiler, a program that translates human-readable code into machine language. Her work democratized programming, allowing non-experts to write code in a language closer to English. This was a crucial step in making computing accessible beyond the realm of mathematicians and engineers.

As technology advanced, the need for even smaller components led to the integrated circuit (IC). In 1958, Jack Kilby at Texas Instruments and Robert Noyce at Fairchild Semiconductor independently developed the IC, which placed multiple transistors on a single chip of semiconductor material. This invention paved the way for the microprocessor, the "brain" of the modern computer. The IC allowed for the miniaturization of electronics, reducing the size of computers from room-sized giants to cabinet-sized machines, and eventually to desktop devices.

### 5. The Microprocessor and the Personal Computer

The invention of the microprocessor in the early 1970s changed everything. A microprocessor is a single integrated circuit that contains the entire CPU of a computer. The Intel 4004, released in 1971, was the first commercial microprocessor. It contained 2,300 transistors and could perform basic arithmetic and logic functions. This tiny chip held the potential to revolutionize society by bringing computing power to individuals.

In the mid-1970s, the personal computer (PC) began to emerge. Hobbyists and engineers, inspired by publications like *Popular Electronics*, built their own computers from kits. The Altair 8800, released in 1975, was one of the first successful personal computers, though it required users to enter programs via toggle switches.

However, it was Apple Computer, founded by Steve Jobs and Steve Wozniak, that popularized the PC for the general public. The Apple II, introduced in 1977, was affordable, user-friendly, and came with a keyboard, monitor, and disk drive. It was designed for home use and education, and it sparked a software revolution. Programs like VisiCalc, the first electronic spreadsheet, demonstrated the practical utility of the PC for business and personal finance.

IBM entered the market in 1981 with the IBM PC, which set an industry standard for hardware architecture. The open architecture of the IBM PC allowed third-party manufacturers to produce compatible hardware and software, leading to an explosion of innovation. Microsoft, led by Bill Gates, provided the operating system (MS-DOS) and later the Windows graphical user interface (GUI), which made computers easier to use through icons, windows, and menus.

The personal computer era transformed society. It democratized access to information, enabling individuals to create, communicate, and solve problems on their own terms. It gave rise to the software industry, the internet, and the digital economy. The computer was no longer a tool for scientists and governments; it was a tool for everyone.

### 6. Connectivity and the Information Age

While the personal computer revolutionized individual productivity, the development of computer networks revolutionized communication. The origins of the internet can be traced back to ARPANET, a project funded by the U.S. Department of Defense in the late 1960s. The goal was to create a communication network that could survive a nuclear attack by decentralizing information.

In 1969, the first four nodes of ARPANET were connected, allowing data to be sent between computers at UCLA, Stanford Research Institute, UC Santa Barbara, and the University of Utah. The key innovation was packet switching, a method of transmitting data in small blocks that could take different
```
