# ctest yolo_bench_pipeline (T10.5) : yolo_bench sur les entrées des trois dumps, en mode
# séquentiel puis --pipeline (deux arènes, deux threads) → mêmes détections, dans le même ordre.
file(MAKE_DIRECTORY ${WORK})
set(ids "")
set(npys "")
foreach(id 000001 000002 000003)
  string(APPEND ids "${id}\n")
  list(APPEND npys ${MODEL}/dumps/${id}/input.npy)
endforeach()
file(WRITE ${WORK}/ids.txt "${ids}")
execute_process(COMMAND ${PYTHON} -c
                "import sys, numpy as np; np.save(sys.argv[1], np.concatenate([np.load(f) for f in sys.argv[2:]]))"
                ${WORK}/inputs.npy ${npys} RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "concaténation des entrées : code ${rc}")
endif()
foreach(mode seq pipeline)
  set(extra "")
  if(mode STREQUAL "pipeline")
    set(extra --pipeline)
  endif()
  execute_process(COMMAND ${BENCH} --model ${MODEL} --inputs ${WORK}/inputs.npy
                          --ids ${WORK}/ids.txt --warmup 0 --conf 0.25 ${extra}
                          --dets ${WORK}/dets_${mode}.jsonl RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "yolo_bench ${mode} : code ${rc}")
  endif()
endforeach()
file(READ ${WORK}/dets_seq.jsonl a)
file(READ ${WORK}/dets_pipeline.jsonl b)
if(NOT a STREQUAL b)
  message(FATAL_ERROR "détections différentes :\n${a}\n${b}")
endif()
