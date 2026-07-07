var path = require("path");
var fs = require("fs");
var rl = require("readline");

var __dirname = path.dirname(fs.realpathSync(__filename));

function assembleWASM(count) {
    var buffers = [];
    for (var i = 0; i < count; ++i)
        buffers.push(fs.readFileSync(path.join(__dirname, "..", "torch-4-part-" + i + ".wasm")));
    return Buffer.concat(buffers);
}

var workerPath = path.join(__dirname, "torch.worker.js");

var engine = {
    locateFile: function(p) {
        if (p.indexOf(".wasm") > -1) return p;
        if (p.indexOf("worker") > -1) return workerPath;
        return p;
    },
    wasmBinary: assembleWASM(6),
    listener: function(line) { process.stdout.write(line + "\n"); },
};

var initEngine = require("./torch-4.js");
initEngine()(engine).then(function(inst) {
    var sendCmd = inst.cwrap("command", null, ["string"]);
    rl.createInterface({ input: process.stdin, terminal: false })
        .on("line", function(line) {
            if (!line) return;
            sendCmd(line);
            if (line === "quit" || line === "exit") process.exit();
        })
        .on("close", function() { process.exit(); });
}).catch(function(e) {
    process.stderr.write("Torch init error: " + e.message + "\n");
    process.exit(1);
});
