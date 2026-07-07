var path = require("path");
var fs = require("fs");
var nodeWorkerThreads = require("worker_threads");
var parentPort = nodeWorkerThreads.parentPort;

Object.defineProperty(nodeWorkerThreads, "isMainThread", { get: function () { return true; }, configurable: true });

var INIT_ENGINE = require("./torch-4.js");

Object.defineProperty(nodeWorkerThreads, "isMainThread", { get: function () { return false; }, configurable: true });

parentPort.removeAllListeners("message");

var Module = {};

Object.assign(global, {
    self: global,
    require: require,
    Module: Module,
    location: { href: __filename },
    Worker: nodeWorkerThreads.Worker,
    importScripts: function (f) { (0, eval)(fs.readFileSync(f, "utf8")) },
    postMessage: function (msg) { parentPort.postMessage(msg) },
    performance: global.performance || { now: function () { return Date.now() } },
});

function threadPrintErr() {
    var text = Array.prototype.slice.call(arguments).join(" ");
    fs.writeSync(2, text + "\n");
}

function threadAlert() {
    var text = Array.prototype.slice.call(arguments).join(" ");
    postMessage({ cmd: "alert", text: text, threadId: Module["_pthread_self"]() });
}

var err = threadPrintErr;
self.alert = threadAlert;

parentPort.on("message", function (data) {
    try {
        if (data.cmd === "load") {
            var Torch = INIT_ENGINE();
            var workerConfig = {
                wasmModule: data.wasmModule,
                wasmMemory: data.wasmMemory,
                buffer: data.wasmMemory.buffer,
                ENVIRONMENT_IS_PTHREAD: true,
                instantiateWasm: function (info, receiveInstance) {
                    var instance = new WebAssembly.Instance(data.wasmModule, info);
                    receiveInstance(instance);
                    return instance.exports;
                },
            };
            Torch(workerConfig).then(function (instance) {
                Module = instance;
            });
        } else if (data.cmd === "run") {
            Module["__performance_now_clock_drift"] = performance.now() - data.time;
            Module["__emscripten_thread_init"](data.threadInfoStruct, 0, 0, 1);
            Module["establishStackSpace"]();
            Module["PThread"].receiveObjectTransfer(data);
            Module["PThread"].threadInit();
            try {
                var result = Module["invokeEntryPoint"](data.start_routine, data.arg);
                if (Module["keepRuntimeAlive"]()) {
                    Module["PThread"].setExitStatus(result);
                } else {
                    Module["__emscripten_thread_exit"](result);
                }
            } catch (ex) {
                if (ex !== "unwind") {
                    if (ex instanceof Module["ExitStatus"]) {
                        if (!Module["keepRuntimeAlive"]()) Module["__emscripten_thread_exit"](ex.status);
                    } else throw ex;
                }
            }
        } else if (data.cmd === "cancel") {
            if (Module["_pthread_self"]()) Module["__emscripten_thread_exit"](-1);
        }
    } catch (ex) {
        err("worker exception: " + ex);
    }
});
