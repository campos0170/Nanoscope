import { pathToFileURL } from 'node:url';
const { marked } = await import(pathToFileURL(process.argv[2]).href);
let input = '';
for await (const chunk of process.stdin) input += chunk;
const { source } = JSON.parse(input);
const html = marked.parse(source, { gfm: true, breaks: false, async: false });
process.stdout.write(JSON.stringify({ html }));
