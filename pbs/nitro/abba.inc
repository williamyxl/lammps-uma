# ABBA slot expansion for the nitro harness.
# Spec lines: LABEL TAG [artifact=PATH] [KEY=VAL ...]
# The first arm is BASE. Each later arm B becomes the block BASE B B BASE.
# Arms are kept in spec order, so list risky arms last.
# shellcheck shell=bash

nitro_abba_read_spec() {
  local spec="$1"
  NITRO_ABBA_N=0
  local line label tag rest tok art env
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    read -r label tag rest <<<"${line}"
    [[ -n "${label}" && -n "${tag}" ]] || nitro_die "bad spec line: ${line}"
    art=""
    env=""
    # shellcheck disable=SC2086
    for tok in ${rest}; do
      if [[ "${tok}" == artifact=* ]]; then
        art="${tok#artifact=}"
      else
        [[ "${tok}" == *=* ]] || nitro_die "bad spec token '${tok}' (want KEY=VAL or artifact=PATH)"
        env="${env}${env:+ }${tok}"
      fi
    done
    NITRO_ABBA_LABEL[NITRO_ABBA_N]="${label}"
    NITRO_ABBA_TAG[NITRO_ABBA_N]="${tag}"
    NITRO_ABBA_ART[NITRO_ABBA_N]="${art}"
    NITRO_ABBA_ENV[NITRO_ABBA_N]="${env}"
    NITRO_ABBA_N=$((NITRO_ABBA_N + 1))
  done < "${spec}"
  [[ "${NITRO_ABBA_N}" -ge 2 ]] || nitro_die "spec needs a BASE arm and at least one treatment"
}

# Fill NITRO_SLOT_* in ABBA order. Index 0..NITRO_SLOT_N-1.
nitro_abba_expand() {
  nitro_abba_read_spec "$1"
  NITRO_SLOT_N=0
  local i
  for ((i = 1; i < NITRO_ABBA_N; i++)); do
    nitro_abba_push 0
    nitro_abba_push "${i}"
    nitro_abba_push "${i}"
    nitro_abba_push 0
  done
}

nitro_abba_push() {
  local src="$1" dst="${NITRO_SLOT_N}"
  NITRO_SLOT_LABEL[dst]="${NITRO_ABBA_LABEL[src]}"
  NITRO_SLOT_TAG[dst]="${NITRO_ABBA_TAG[src]}"
  NITRO_SLOT_ART[dst]="${NITRO_ABBA_ART[src]}"
  NITRO_SLOT_ENVSTR[dst]="${NITRO_ABBA_ENV[src]}"
  NITRO_SLOT_N=$((NITRO_SLOT_N + 1))
}

nitro_abba_write_skeleton() {
  local spec="$1" out="$2"
  nitro_abba_expand "${spec}"
  mkdir -p "${out}"
  : > "${out}/slot_order.txt"
  local i name dir
  for ((i = 0; i < NITRO_SLOT_N; i++)); do
    printf '%d\t%s\t%s\t%s\t%s\n' \
      "${i}" "${NITRO_SLOT_LABEL[i]}" "${NITRO_SLOT_TAG[i]}" \
      "${NITRO_SLOT_ENVSTR[i]}" "${NITRO_SLOT_ART[i]}" >> "${out}/slot_order.txt"
    name=$(printf '%02d_%s' "${i}" "${NITRO_SLOT_LABEL[i]}")
    dir="${out}/slots/${name}"
    mkdir -p "${dir}"
    printf '%s\n' "${NITRO_SLOT_LABEL[i]}" > "${dir}/label"
    printf '%s\n' "${NITRO_SLOT_ENVSTR[i]}" > "${dir}/arm.env"
    cat > "${dir}/meta.json" <<EOF
{
  "dry_run": true,
  "slot": ${i},
  "label": "${NITRO_SLOT_LABEL[i]}",
  "tag": "${NITRO_SLOT_TAG[i]}",
  "artifact": "${NITRO_SLOT_ART[i]}",
  "env": "${NITRO_SLOT_ENVSTR[i]}",
  "nitro_flags": null,
  "hosts": []
}
EOF
  done
  echo "ABBA_DRY slots=${NITRO_SLOT_N} ${out}/slot_order.txt"
}

# Reuse pbs/bin/<sha7>/lmp when its sha256 matches the sidecar. Otherwise build
# that tag in a detached worktree. Does not check out the main worktree.
nitro_ensure_lmp() {
  local tag="$1"
  local sha sha7 dest bin src got want
  sha=$(git -C "${REL}" rev-parse -q --verify "${tag}^{commit}") \
    || nitro_die "arm tag ${tag} does not exist"
  sha7=$(git -C "${REL}" rev-parse --short=7 "${sha}")
  dest="${REL}/pbs/bin/${sha7}"
  bin="${dest}/lmp"
  if [[ -x "${bin}" && -f "${bin}.sha256" ]]; then
    got=$(sha256sum "${bin}" | awk '{print $1}')
    want=$(awk '{print $1}' "${bin}.sha256")
    if [[ "${got}" == "${want}" ]]; then
      echo "LMP_REUSE ${bin}"
      LMP="${bin}"
      if [[ -f "${REL}/pbs/out/.worktrees/${sha7}/cmake/CMakeLists.txt" ]]; then
        NITRO_LAST_SRC="${REL}/pbs/out/.worktrees/${sha7}"
      else
        NITRO_LAST_SRC="${REL}"
      fi
      return 0
    fi
    echo "LMP_SHA_MISMATCH ${bin}"
  fi
  src="${REL}/pbs/out/.worktrees/${sha7}"
  if [[ ! -f "${src}/cmake/CMakeLists.txt" ]]; then
    rm -rf "${src}"
    git -C "${REL}" worktree add --detach "${src}" "${sha}" \
      || nitro_die "worktree add failed for ${tag}"
  fi
  NITRO_SRC="${src}"
  NITRO_LAST_SRC="${src}"
  NITRO_BUILD="${REL}/build-nitro/${sha7}"
  nitro_cmake_build
  mkdir -p "${dest}"
  cp -f "${LMP}" "${bin}"
  sha256sum "${bin}" | awk '{print $1}' > "${bin}.sha256"
  LMP="${bin}"
  echo "LMP_BUILT ${bin}"
}
