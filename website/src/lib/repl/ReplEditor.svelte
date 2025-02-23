<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import type * as Monaco from "monaco-editor/esm/vs/editor/editor.api";

  let containerEl: HTMLDivElement;
  let editor: Monaco.editor.IStandaloneCodeEditor;
  let monaco: typeof Monaco;

  onMount(async () => {
    monaco = (await import("$lib/monaco")).default;

    editor = monaco.editor.create(containerEl, { fontSize: 14 });
    const model = monaco.editor.createModel(
      `import { grad, numpy as np } from "@jax-js/core";

const f = (x: np.Array) => x.mul(x);
const df = grad(f);

const x = np.array([1, 2, 3]);
console.log(f(x).js());
console.log(df(x).js());
`,
      "typescript",
      monaco.Uri.parse("file:///main.ts"),
    );
    editor.setModel(model);
  });

  onDestroy(() => {
    monaco?.editor.getModels().forEach((model) => model.dispose());
    editor?.dispose();
  });
</script>

<div class="h-full" bind:this={containerEl}></div>
