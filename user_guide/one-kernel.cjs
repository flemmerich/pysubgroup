// Jupyter Book 2.1.7's execution semaphore ignores --execute-parallel.
// Its CPU-count-minus-one default is limited here to one kernel for this process.
const os = require('node:os');
const cpus = os.cpus();
os.cpus = () => cpus.slice(0, 2);
