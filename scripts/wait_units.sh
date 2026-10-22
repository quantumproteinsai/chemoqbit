#!/bin/sh
# Block while any of the given systemd --user units is busy. A unit waiting to be
# restarted (ActiveState=activating, SubState=auto-restart) counts as busy, so a
# queued job does not slip in during a 2-minute restart pause.
busy() {
    for u in "$@"; do
        case "$(systemctl --user show -p ActiveState --value "$u")" in
            active|activating|deactivating|reloading) return 0 ;;
        esac
    done
    return 1
}
while busy "$@"; do sleep 60; done
