# =============================================================================
# _quota_preflight.sh — 项目配额预检共享库(v2.4.4,source 使用)
#
# [P0-1 根因] df 查的是整个 Lustre 文件系统(4.3 PiB / 百亿 inode),不是项目
# 配额,必然假放行 —— 一切判决必须走 lquota。本函数被三处 source:
#   qsub_corrected_rerun.sh(全量编排,默认 300GiB/20000)
#   qsub_resisc45.sh / 40_build_resisc45.pbs(泛化 tier,默认 60GiB/20000)
#
# 用法: quota_preflight <TARGET_DIR> [MIN_GIB] [MIN_INODES]
#   项目号默认从 TARGET /scratch/<proj>/... 解析,PROJECT= 覆盖;
#   MIN_QUOTA_GIB / MIN_QUOTA_INODES env 优先于位置参数。
#   lquota 列位按 2026-07-24 用户实测:$1=项目 $2=fs,$3-$5 空间
#   used/quota/limit(带单位),$9-$11 文件数 used/quota/limit;
#   排版不符会硬错(把 lquota 原始输出发回适配,勿放宽)。
# =============================================================================

quota_preflight() {
  local target="$1"
  local min_gib="${MIN_QUOTA_GIB:-${2:-300}}"
  local min_inodes="${MIN_QUOTA_INODES:-${3:-20000}}"
  mkdir -p "$target"
  local proj="${PROJECT:-$(echo "$target" | sed -n 's,^/scratch/\([^/]*\)/.*,\1,p')}"
  [ -n "$proj" ] || { echo "FATAL: 无法从 $target 解析项目号(非 /scratch/<proj>/ 路径?),请设 PROJECT=" >&2; return 1; }

  local LQ SUSED SQUOTA SLIMIT IUSED IQUOTA ILIMIT
  LQ=$(lquota 2>/dev/null | awk -v p="$proj" '$1 == p && $2 == "scratch"') || true
  [ -n "$LQ" ] || { echo "FATAL: 无法从 lquota 读取 $proj scratch 配额(lquota 不可用或无该项目行)。" >&2; return 1; }
  read -r _ _ SUSED SQUOTA SLIMIT _ _ _ IUSED IQUOTA ILIMIT _ <<<"$LQ"

  _to_gib() {
    echo "$1" | awk '{
      v=$0; unit="";
      if (match(v, /[a-zA-Z]+$/)) { unit=tolower(substr(v, RSTART)); v=substr(v, 1, RSTART-1); }
      m=1
      if (unit=="kb"||unit=="kib") m=1.0/1048576;
      if (unit=="mb"||unit=="mib") m=1.0/1024;
      if (unit=="gb"||unit=="gib") m=1.0;
      if (unit=="tb"||unit=="tib") m=1024.0;
      if (unit=="b"||unit=="") m=1.0/1073741824;
      printf "%.1f", v*m
    }'
  }
  local SUSED_G SQUOTA_G SREMAIN_G IREMAIN
  SUSED_G=$(_to_gib "$SUSED"); SQUOTA_G=$(_to_gib "$SQUOTA")
  SREMAIN_G=$(awk -v q="$SQUOTA_G" -v u="$SUSED_G" 'BEGIN{printf "%.1f", q-u}')
  IREMAIN=$((IQUOTA - IUSED))

  echo "== 预检 项目 $proj (scratch,lquota): 空间 used=$SUSED quota=$SQUOTA → 剩余 ${SREMAIN_G} GiB;文件数 used=$IUSED quota=$IQUOTA → 剩余 $IREMAIN(阈值 ${min_gib} GiB / ${min_inodes})"
  echo "   (对照 df —— 仅供参考,那是整个文件系统不是配额: $(df -h "$target" 2>/dev/null | awk 'NR==2{print $4" free"}'))"

  if awk -v r="$SREMAIN_G" -v m="$min_gib" 'BEGIN{exit !(r < m)}'; then
    echo "FATAL: 项目 $proj 空间配额剩余 ${SREMAIN_G} GiB < ${min_gib} GiB —— 清理旧产物或换配额充足的项目。" >&2; return 1
  fi
  if [ "$IREMAIN" -lt "$min_inodes" ]; then
    echo "FATAL: 项目 $proj 文件数配额剩余 $IREMAIN < $min_inodes —— 清理零散小文件或换项目。" >&2; return 1
  fi
  return 0
}
