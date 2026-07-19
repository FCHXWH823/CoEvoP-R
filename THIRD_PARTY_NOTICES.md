# Third-Party Notices

CoEvoP&R interoperates with external projects that are not vendored in this
repository. Users must obtain those projects and benchmark data separately and
comply with their upstream licenses.

## OpenEvolve

The persistent program database, archive-conditioned prompting, island
populations, migration, and MAP-Elites organization are adapted from concepts
in OpenEvolve:

- Project: <https://github.com/algorithmicsuperintelligence/openevolve>
- License: Apache License 2.0

CoEvoP&R implements these mechanisms for a restricted placement-objective DSL;
it does not execute arbitrary model-generated Python and does not vendor the
OpenEvolve repository. Additional detail appears in
`docs/OPENEVOLVE_ATTRIBUTION.md`.

## DREAMPlace

The patch files under `patches/dreamplace/` modify DREAMPlace integration
points and include contextual DREAMPlace source lines:

- Project: <https://github.com/limbo018/DREAMPlace>
- License: BSD 3-Clause License

The upstream license notice is:

```text
BSD 3-Clause License

Copyright (c) 2019,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Other External Backends And Data

OpenROAD, ChiPBench, CircuitNet, OpenTimer, technology libraries, standard-cell
libraries, and benchmark suites remain external. This repository provides
adapters and public path templates only. Their source code, binaries, netlists,
libraries, and physical-design files are not redistributed by the anonymous
source-archive builder.
