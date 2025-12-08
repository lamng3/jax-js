<script lang="ts">
  import {
    ChevronRightIcon,
    ImageIcon,
    InfoIcon,
    TriangleAlertIcon,
    XIcon,
  } from "@lucide/svelte";

  import type { ConsoleLine } from "./runner.svelte";

  let { line, showTime = false }: { line: ConsoleLine; showTime?: boolean } =
    $props();
</script>

<div
  class={[
    "py-0.5 border-t flex items-start gap-x-2",
    line.level === "error"
      ? "border-red-200 bg-red-50"
      : line.level === "warn"
        ? "border-yellow-200 bg-yellow-50"
        : "border-gray-200",
  ]}
>
  {#if line.level === "log"}
    <ChevronRightIcon size={18} class="shrink-0 text-gray-300" />
  {:else if line.level === "info"}
    <InfoIcon size={18} class="shrink-0 text-blue-500" />
  {:else if line.level === "warn"}
    <TriangleAlertIcon size={18} class="shrink-0 text-yellow-500" />
  {:else if line.level === "error"}
    <XIcon size={18} class="shrink-0 text-red-500" />
  {:else if line.level === "image"}
    <ImageIcon size={18} class="shrink-0 text-gray-400" />
  {/if}
  <p class="font-mono whitespace-pre-wrap">
    {#if line.level === "image"}
      <img
        src={line.data[0]}
        alt="Output from displayImage()"
        class="max-w-full my-0.5"
      />
    {:else}
      {line.data.join(" ")}
    {/if}
  </p>
  {#if showTime}
    <p
      class="hidden md:block ml-auto shrink-0 font-mono text-gray-400 select-none"
    >
      {new Date(line.time).toLocaleTimeString()}
    </p>
  {/if}
</div>
