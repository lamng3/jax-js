<script lang="ts">
  import ReplEditor from "$lib/repl/ReplEditor.svelte";
</script>

<main class="flex h-dvh">
  <div class="shrink-0 border">sidebar</div>
  <div class="flex-1 flex flex-col">
    <div class="border">hello</div>
    <div class="flex-1">
      <ReplEditor />
    </div>
    <div class="border">hello</div>
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";
</style>
