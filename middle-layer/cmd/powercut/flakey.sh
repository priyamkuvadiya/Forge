#!/bin/bash
# Power-cut test for the session store, with dm-flakey (as xfstests does it).
# The database lives on a filesystem over a flakey device. At a random moment the
# writer is frozen, everything it was promised so far is noted, and the device
# starts dropping every write. The writer is killed and the filesystem unmounted:
# whatever was only in the page cache never reaches the disk, as in a power cut.
# Remount with writes working, and check every promise.
#
# Needs root, Linux with the dm_flakey module, and a Linux build of this command:
#   GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o powercut ./cmd/powercut
#   sudo modprobe dm_flakey && sudo bash cmd/powercut/flakey.sh ./powercut 10 0
# UNSAFE=1 is the control: SQLite stops syncing, and the cut must then lose data,
# or this script is not cutting anything.
# usage: flakey.sh BINARY ROUNDS UNSAFE(0|1)
set -u
BINARY=$1; ROUNDS=$2; UNSAFE=$3
W=/root/flakey; rm -rf $W; mkdir -p $W/mnt
cp "$BINARY" $W/powercut; chmod +x $W/powercut
truncate -s 2G $W/disk.img
LOOP=$(losetup -f --show $W/disk.img)
SIZE=$(blockdev --getsz $LOOP)
up="0 $SIZE flakey $LOOP 0 3600 0"
down="0 $SIZE flakey $LOOP 0 0 3600 1 drop_writes"
dmsetup create pc --table "$up"
mkfs.ext4 -q /dev/mapper/pc
mount /dev/mapper/pc $W/mnt

fail=0
for r in $(seq 1 $ROUNDS); do
  POWERCUT_UNSAFE=$UNSAFE $W/powercut write $W/mnt/test.db > $W/acks_$r 2>$W/err_$r &
  pid=$!
  sleep $(awk -v s=$RANDOM 'BEGIN{print 2+(s%5000)/1000}')
  kill -STOP $pid
  promised=$(wc -l < $W/acks_$r)
  head -n "$promised" $W/acks_$r > $W/promised_$r
  # The cut: from here no write reaches the disk. --nolockfs, or suspending would
  # freeze the filesystem and flush the page cache first.
  dmsetup suspend --nolockfs pc; dmsetup load pc --table "$down"; dmsetup resume pc
  kill -9 $pid; wait $pid 2>/dev/null
  umount $W/mnt
  dmsetup suspend --nolockfs pc; dmsetup load pc --table "$up"; dmsetup resume pc
  mount /dev/mapper/pc $W/mnt
  echo "round $r: $promised promises before the cut"
  $W/powercut check $W/mnt/test.db $W/promised_$r || fail=1
done

sync; umount $W/mnt || umount -l $W/mnt
dmsetup remove pc || dmsetup remove --force pc
losetup -d $LOOP; rm -rf $W
exit $fail
